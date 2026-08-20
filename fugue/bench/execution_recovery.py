from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.bench.execution import (
    HARBOR_PARENT_RUNNER_CLASSIFIER_DIGEST,
    CellOutcome,
    PlannedCell,
    execute_cells,
    record_recovered_cell_outcome,
)
from fugue.bench.files import atomic_write_json
from fugue.bench.harbor_outcome import HARBOR_TERMINAL_CLASSIFIER_DIGEST
from fugue.redaction import redact_text, redact_value, secrets_from_env

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")

TerminalKind = Literal[
    "success",
    "task_failure",
    "agent_timeout",
    "cancelled",
    "runner_start_failure",
    "sandbox_lost",
    "transport_interrupted",
    "routing_failure",
    "identity_drift",
    "privacy_failure",
    "evidence_failure",
    "cleanup_failure",
    "execution_failure",
]
EventKind = Literal[
    "controller_initialized",
    "stage_authorized",
    "physical_execution_leased",
    "physical_execution_started",
    "physical_execution_terminal",
    "physical_execution_cleanup",
    "physical_execution_cost",
    "canonical_result_selected",
    "fatal_integrity_halt",
]

_RETRYABLE_INFRASTRUCTURE = frozenset(
    {"runner_start_failure", "sandbox_lost", "transport_interrupted"}
)
_BEHAVIORAL_TERMINAL = frozenset(
    {"success", "task_failure", "agent_timeout", "cancelled"}
)
_FATAL_INTEGRITY = frozenset(
    {
        "routing_failure",
        "identity_drift",
        "privacy_failure",
        "evidence_failure",
        "cleanup_failure",
        "execution_failure",
    }
)
_TERMINAL_KINDS = _RETRYABLE_INFRASTRUCTURE | _BEHAVIORAL_TERMINAL | _FATAL_INTEGRITY
_EVENT_PAYLOAD_FIELDS: dict[str, set[str]] = {
    "controller_initialized": {"schedule"},
    "stage_authorized": {"authorization"},
    "physical_execution_leased": {
        "identity",
        "reserve_micro_usd",
        "admission_weight",
        "planned_harbor_resources",
    },
    "physical_execution_started": {"observed_harbor_resources"},
    "physical_execution_terminal": {
        "terminal_kind",
        "result_reference",
        "cell_outcome",
    },
    "physical_execution_cleanup": {
        "verified",
        "scope_verified",
        "post_run_inventory",
        "discovered_resources",
        "inspected_resources",
        "removed_resources",
        "remaining_resources",
        "receipt_reference",
    },
    "physical_execution_cost": {
        "actual_cost_micro_usd",
        "authoritative",
        "lease_exceeded",
        "source",
        "receipt_reference",
    },
    "canonical_result_selected": {
        "result_reference",
        "reconciliation",
    },
    "fatal_integrity_halt": {"reason"},
}

LOCAL_HARBOR_RECOVERY_ADAPTER_CONTRACT = stable_digest(
    {
        "contract": "fugue.execution-recovery-adapters",
        "version": 4,
        "harbor_terminal_classifier": HARBOR_TERMINAL_CLASSIFIER_DIGEST,
        "parent_runner_terminal_classifier": (
            HARBOR_PARENT_RUNNER_CLASSIFIER_DIGEST
        ),
        "cost": "authoritative-receipt",
        "cleanup": "exact-scope-post-run-inventory",
        "interrupted": (
            "parent-process-or-harbor-terminal-hook-and-result-reconciliation"
        ),
        "physical_namespace": (
            "unique-harbor-config-job-result-and-agent-state-per-execution"
        ),
    }
)


class ExecutionRecoveryError(RuntimeError):
    """A durable controller cannot safely admit or reconcile execution."""


class ExecutionRecoveryPaused(ExecutionRecoveryError):
    """Admission paused without invalidating or rerunning completed work."""


class ExecutionFinalizationPending(ExecutionRecoveryPaused):
    """Host-only finalization may be retried without rerunning the Agent.

    Finalizers must raise this type only when already-written Agent work is
    intact and the remaining operation is safe and idempotent. Integrity or
    identity disagreements must use another exception and remain fatal.
    """


class UnsupportedHostedFinalizationRecovery(ExecutionRecoveryError):
    """A hosted Evaluation cannot resume without replacing its live graph.

    This is a terminal integrity condition, not a retryable pause.  The
    canonical local attempt remains authoritative, but the installed hosted
    backend cannot safely continue the exact Evaluation lifecycle after the
    owning process is lost.
    """


class ExecutionJournalCorrupt(ExecutionRecoveryError):
    """The durable execution event chain is malformed or was altered."""


@dataclass(frozen=True)
class LogicalExecutionPlanV1:
    """Execution-only policy for one existing canonical Fugue attempt."""

    schema_version: Literal[1]
    logical_attempt_id: str
    stage_id: str
    stage_ordinal: int
    admission_block_id: str
    block_ordinal: int
    attempt_ordinal: int
    admission_weight: int
    maximum_cost_micro_usd: int
    planned_harbor_resources: tuple[str, ...] = ()
    plan_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported logical execution plan schema")
        _require_digest(self.logical_attempt_id, "logical attempt id")
        _require_safe_id(self.stage_id, "stage id")
        _require_safe_id(self.admission_block_id, "admission block id")
        for value, label in (
            (self.stage_ordinal, "stage ordinal"),
            (self.block_ordinal, "block ordinal"),
            (self.attempt_ordinal, "attempt ordinal"),
            (self.maximum_cost_micro_usd, "maximum cost"),
        ):
            _require_nonnegative_int(value, label)
        _require_positive_int(self.admission_weight, "admission weight")
        if len(set(self.planned_harbor_resources)) != len(
            self.planned_harbor_resources
        ):
            raise ValueError("planned Harbor resources must be unique")
        for resource in self.planned_harbor_resources:
            _require_safe_id(resource, "planned Harbor resource")
        digest = stable_digest(self._unsigned())
        if self.plan_digest and self.plan_digest != digest:
            raise ValueError("logical execution plan digest does not match")
        object.__setattr__(self, "plan_digest", digest)

    @classmethod
    def create(
        cls,
        *,
        logical_attempt_id: str,
        stage_id: str,
        stage_ordinal: int,
        admission_block_id: str,
        block_ordinal: int,
        attempt_ordinal: int,
        admission_weight: int,
        maximum_cost_micro_usd: int,
        planned_harbor_resources: Sequence[str] = (),
    ) -> LogicalExecutionPlanV1:
        return cls(
            1,
            logical_attempt_id,
            stage_id,
            stage_ordinal,
            admission_block_id,
            block_ordinal,
            attempt_ordinal,
            admission_weight,
            maximum_cost_micro_usd,
            tuple(planned_harbor_resources),
        )

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_attempt_id": self.logical_attempt_id,
            "stage_id": self.stage_id,
            "stage_ordinal": self.stage_ordinal,
            "admission_block_id": self.admission_block_id,
            "block_ordinal": self.block_ordinal,
            "attempt_ordinal": self.attempt_ordinal,
            "admission_weight": self.admission_weight,
            "maximum_cost_micro_usd": self.maximum_cost_micro_usd,
            "planned_harbor_resources": list(self.planned_harbor_resources),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LogicalExecutionPlanV1:
        _require_fields(
            raw,
            {
                "schema_version",
                "logical_attempt_id",
                "stage_id",
                "stage_ordinal",
                "admission_block_id",
                "block_ordinal",
                "attempt_ordinal",
                "admission_weight",
                "maximum_cost_micro_usd",
                "planned_harbor_resources",
                "plan_digest",
            },
            "logical execution plan",
        )
        return cls(
            schema_version=_strict_int(raw["schema_version"], "schema_version"),  # type: ignore[arg-type]
            logical_attempt_id=_strict_str(
                raw["logical_attempt_id"], "logical_attempt_id"
            ),
            stage_id=_strict_str(raw["stage_id"], "stage_id"),
            stage_ordinal=_strict_int(raw["stage_ordinal"], "stage_ordinal"),
            admission_block_id=_strict_str(
                raw["admission_block_id"], "admission_block_id"
            ),
            block_ordinal=_strict_int(raw["block_ordinal"], "block_ordinal"),
            attempt_ordinal=_strict_int(raw["attempt_ordinal"], "attempt_ordinal"),
            admission_weight=_strict_int(raw["admission_weight"], "admission_weight"),
            maximum_cost_micro_usd=_strict_int(
                raw["maximum_cost_micro_usd"], "maximum_cost_micro_usd"
            ),
            planned_harbor_resources=tuple(
                _strict_str(item, "planned_harbor_resource")
                for item in _strict_list(
                    raw["planned_harbor_resources"], "planned_harbor_resources"
                )
            ),
            plan_digest=_strict_str(raw["plan_digest"], "plan_digest"),
        )


@dataclass(frozen=True)
class ExecutionScheduleV1:
    """Digest-bound weighted admission and spend policy for a logical matrix."""

    schema_version: Literal[1]
    schedule_id: str
    logical_attempts: tuple[LogicalExecutionPlanV1, ...]
    capacity_units: int
    maximum_in_flight_executions: int
    maximum_physical_executions: int
    maximum_infrastructure_replacements: int
    maximum_total_micro_usd: int
    maximum_in_flight_micro_usd: int
    adapter_contract_digest: str
    retryable_terminal_kinds: tuple[str, ...]
    fatal_terminal_kinds: tuple[str, ...]
    schedule_digest: str = ""

    def __post_init__(self) -> None:  # noqa: C901 - one strict lock boundary
        if self.schema_version != 1:
            raise ValueError("unsupported execution schedule schema")
        _require_safe_id(self.schedule_id, "schedule id")
        if not self.logical_attempts:
            raise ValueError("execution schedule requires logical attempts")
        logical_ids = [item.logical_attempt_id for item in self.logical_attempts]
        if len(set(logical_ids)) != len(logical_ids):
            raise ValueError("logical attempts must be unique")
        order = [
            (
                item.stage_ordinal,
                item.block_ordinal,
                item.attempt_ordinal,
                item.logical_attempt_id,
            )
            for item in self.logical_attempts
        ]
        if order != sorted(order):
            raise ValueError("logical attempts must be in canonical admission order")
        stage_names: dict[int, str] = {}
        block_names: dict[tuple[int, int], str] = {}
        stage_block_ordinals: dict[int, set[int]] = {}
        block_attempt_ordinals: dict[tuple[int, int], list[int]] = {}
        for attempt in self.logical_attempts:
            prior_stage = stage_names.setdefault(
                attempt.stage_ordinal, attempt.stage_id
            )
            if prior_stage != attempt.stage_id:
                raise ValueError("one stage ordinal cannot name multiple stages")
            block_key = (attempt.stage_ordinal, attempt.block_ordinal)
            stage_block_ordinals.setdefault(attempt.stage_ordinal, set()).add(
                attempt.block_ordinal
            )
            prior_block = block_names.setdefault(block_key, attempt.admission_block_id)
            if prior_block != attempt.admission_block_id:
                raise ValueError("one block ordinal cannot name multiple blocks")
            block_attempt_ordinals.setdefault(block_key, []).append(
                attempt.attempt_ordinal
            )
            if attempt.admission_weight > self.capacity_units:
                raise ValueError("logical attempt exceeds admission capacity")
            if attempt.maximum_cost_micro_usd > self.maximum_in_flight_micro_usd:
                raise ValueError("logical attempt exceeds in-flight spend ceiling")
        if sorted(stage_names) != list(range(len(stage_names))):
            raise ValueError("stage ordinals must be contiguous from zero")
        if len(set(stage_names.values())) != len(stage_names):
            raise ValueError("stage ids must be unique")
        if len(set(block_names.values())) != len(block_names):
            raise ValueError("admission block ids must be unique")
        if any(
            sorted(ordinals) != list(range(len(ordinals)))
            for ordinals in stage_block_ordinals.values()
        ):
            raise ValueError("block ordinals must be contiguous within each stage")
        for ordinals in block_attempt_ordinals.values():
            if ordinals != list(range(len(ordinals))):
                raise ValueError(
                    "attempt ordinals must be contiguous within each block"
                )
        _require_positive_int(self.capacity_units, "capacity units")
        _require_positive_int(
            self.maximum_in_flight_executions,
            "maximum in-flight executions",
        )
        _require_positive_int(
            self.maximum_physical_executions,
            "maximum physical executions",
        )
        _require_nonnegative_int(
            self.maximum_infrastructure_replacements,
            "maximum infrastructure replacements",
        )
        expected_physical = (
            len(self.logical_attempts) + self.maximum_infrastructure_replacements
        )
        if self.maximum_physical_executions != expected_physical:
            raise ValueError(
                "physical execution ceiling must equal logical attempts plus "
                "infrastructure replacements"
            )
        _require_positive_int(self.maximum_total_micro_usd, "maximum total cost")
        _require_positive_int(
            self.maximum_in_flight_micro_usd,
            "maximum in-flight cost",
        )
        if self.maximum_in_flight_micro_usd > self.maximum_total_micro_usd:
            raise ValueError("in-flight cost ceiling exceeds total cost ceiling")
        _require_digest(self.adapter_contract_digest, "adapter contract digest")
        planned = sum(item.maximum_cost_micro_usd for item in self.logical_attempts)
        if planned > self.maximum_total_micro_usd:
            raise ValueError("planned logical attempts exceed total cost ceiling")
        if set(self.retryable_terminal_kinds) - _RETRYABLE_INFRASTRUCTURE:
            raise ValueError("only typed infrastructure failures may be retried")
        if len(set(self.retryable_terminal_kinds)) != len(
            self.retryable_terminal_kinds
        ):
            raise ValueError("retryable terminal kinds must be unique")
        if set(self.fatal_terminal_kinds) - _FATAL_INTEGRITY:
            raise ValueError("fatal terminal kinds must be integrity failures")
        if len(set(self.fatal_terminal_kinds)) != len(self.fatal_terminal_kinds):
            raise ValueError("fatal terminal kinds must be unique")
        digest = stable_digest(self._unsigned())
        if self.schedule_digest and self.schedule_digest != digest:
            raise ValueError("execution schedule digest does not match")
        object.__setattr__(self, "schedule_digest", digest)

    @classmethod
    def create(
        cls,
        *,
        schedule_id: str,
        logical_attempts: Sequence[LogicalExecutionPlanV1],
        capacity_units: int,
        maximum_in_flight_executions: int,
        maximum_physical_executions: int,
        maximum_infrastructure_replacements: int,
        maximum_total_micro_usd: int,
        maximum_in_flight_micro_usd: int,
        adapter_contract_digest: str = LOCAL_HARBOR_RECOVERY_ADAPTER_CONTRACT,
        retryable_terminal_kinds: Sequence[str] = tuple(
            sorted(_RETRYABLE_INFRASTRUCTURE)
        ),
        fatal_terminal_kinds: Sequence[str] = tuple(sorted(_FATAL_INTEGRITY)),
    ) -> ExecutionScheduleV1:
        return cls(
            1,
            schedule_id,
            tuple(logical_attempts),
            capacity_units,
            maximum_in_flight_executions,
            maximum_physical_executions,
            maximum_infrastructure_replacements,
            maximum_total_micro_usd,
            maximum_in_flight_micro_usd,
            adapter_contract_digest,
            tuple(retryable_terminal_kinds),
            tuple(fatal_terminal_kinds),
        )

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "logical_attempts": [item.to_dict() for item in self.logical_attempts],
            "capacity_units": self.capacity_units,
            "maximum_in_flight_executions": self.maximum_in_flight_executions,
            "maximum_physical_executions": self.maximum_physical_executions,
            "maximum_infrastructure_replacements": (
                self.maximum_infrastructure_replacements
            ),
            "maximum_total_micro_usd": self.maximum_total_micro_usd,
            "maximum_in_flight_micro_usd": self.maximum_in_flight_micro_usd,
            "adapter_contract_digest": self.adapter_contract_digest,
            "retryable_terminal_kinds": list(self.retryable_terminal_kinds),
            "fatal_terminal_kinds": list(self.fatal_terminal_kinds),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "schedule_digest": self.schedule_digest}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExecutionScheduleV1:
        _require_fields(
            raw,
            {
                "schema_version",
                "schedule_id",
                "logical_attempts",
                "capacity_units",
                "maximum_in_flight_executions",
                "maximum_physical_executions",
                "maximum_infrastructure_replacements",
                "maximum_total_micro_usd",
                "maximum_in_flight_micro_usd",
                "adapter_contract_digest",
                "retryable_terminal_kinds",
                "fatal_terminal_kinds",
                "schedule_digest",
            },
            "execution schedule",
        )
        return cls(
            schema_version=_strict_int(raw["schema_version"], "schema_version"),  # type: ignore[arg-type]
            schedule_id=_strict_str(raw["schedule_id"], "schedule_id"),
            logical_attempts=tuple(
                LogicalExecutionPlanV1.from_dict(
                    _strict_mapping(item, "logical_attempt")
                )
                for item in _strict_list(raw["logical_attempts"], "logical_attempts")
            ),
            capacity_units=_strict_int(raw["capacity_units"], "capacity_units"),
            maximum_in_flight_executions=_strict_int(
                raw["maximum_in_flight_executions"],
                "maximum_in_flight_executions",
            ),
            maximum_physical_executions=_strict_int(
                raw["maximum_physical_executions"],
                "maximum_physical_executions",
            ),
            maximum_infrastructure_replacements=_strict_int(
                raw["maximum_infrastructure_replacements"],
                "maximum_infrastructure_replacements",
            ),
            maximum_total_micro_usd=_strict_int(
                raw["maximum_total_micro_usd"], "maximum_total_micro_usd"
            ),
            maximum_in_flight_micro_usd=_strict_int(
                raw["maximum_in_flight_micro_usd"],
                "maximum_in_flight_micro_usd",
            ),
            adapter_contract_digest=_strict_str(
                raw["adapter_contract_digest"], "adapter_contract_digest"
            ),
            retryable_terminal_kinds=tuple(
                _strict_str(item, "retryable_terminal_kind")
                for item in _strict_list(
                    raw["retryable_terminal_kinds"],
                    "retryable_terminal_kinds",
                )
            ),
            fatal_terminal_kinds=tuple(
                _strict_str(item, "fatal_terminal_kind")
                for item in _strict_list(
                    raw["fatal_terminal_kinds"], "fatal_terminal_kinds"
                )
            ),
            schedule_digest=_strict_str(raw["schedule_digest"], "schedule_digest"),
        )


@dataclass(frozen=True)
class StageExecutionAuthorizationV1:
    """One approval-bounded stage of an unchanged full execution schedule."""

    schema_version: Literal[1]
    preview_digest: str
    approval_digest: str
    schedule_digest: str
    stage_id: str
    maximum_logical_attempts: int
    maximum_physical_executions: int
    maximum_cost_micro_usd: int
    authorization_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported stage authorization schema")
        for value, label in (
            (self.preview_digest, "preview digest"),
            (self.approval_digest, "approval digest"),
            (self.schedule_digest, "schedule digest"),
        ):
            _require_digest(value, label)
        _require_safe_id(self.stage_id, "stage id")
        _require_positive_int(
            self.maximum_logical_attempts, "stage logical-attempt ceiling"
        )
        _require_positive_int(
            self.maximum_physical_executions, "stage physical-execution ceiling"
        )
        if self.maximum_physical_executions < self.maximum_logical_attempts:
            raise ValueError(
                "stage physical-execution ceiling cannot be below its logical ceiling"
            )
        _require_positive_int(self.maximum_cost_micro_usd, "stage cost ceiling")
        digest = stable_digest(self._unsigned())
        if self.authorization_digest and self.authorization_digest != digest:
            raise ValueError("stage authorization digest does not match")
        object.__setattr__(self, "authorization_digest", digest)

    @classmethod
    def create(
        cls,
        *,
        preview_digest: str,
        approval_digest: str,
        schedule_digest: str,
        stage_id: str,
        maximum_logical_attempts: int,
        maximum_physical_executions: int,
        maximum_cost_micro_usd: int,
    ) -> StageExecutionAuthorizationV1:
        return cls(
            1,
            preview_digest,
            approval_digest,
            schedule_digest,
            stage_id,
            maximum_logical_attempts,
            maximum_physical_executions,
            maximum_cost_micro_usd,
        )

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "preview_digest": self.preview_digest,
            "approval_digest": self.approval_digest,
            "schedule_digest": self.schedule_digest,
            "stage_id": self.stage_id,
            "maximum_logical_attempts": self.maximum_logical_attempts,
            "maximum_physical_executions": self.maximum_physical_executions,
            "maximum_cost_micro_usd": self.maximum_cost_micro_usd,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "authorization_digest": self.authorization_digest}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> StageExecutionAuthorizationV1:
        _require_fields(
            raw,
            {
                "schema_version",
                "preview_digest",
                "approval_digest",
                "schedule_digest",
                "stage_id",
                "maximum_logical_attempts",
                "maximum_physical_executions",
                "maximum_cost_micro_usd",
                "authorization_digest",
            },
            "stage execution authorization",
        )
        return cls(
            schema_version=_strict_int(raw["schema_version"], "schema_version"),  # type: ignore[arg-type]
            preview_digest=_strict_str(raw["preview_digest"], "preview_digest"),
            approval_digest=_strict_str(raw["approval_digest"], "approval_digest"),
            schedule_digest=_strict_str(raw["schedule_digest"], "schedule_digest"),
            stage_id=_strict_str(raw["stage_id"], "stage_id"),
            maximum_logical_attempts=_strict_int(
                raw["maximum_logical_attempts"], "maximum_logical_attempts"
            ),
            maximum_physical_executions=_strict_int(
                raw["maximum_physical_executions"], "maximum_physical_executions"
            ),
            maximum_cost_micro_usd=_strict_int(
                raw["maximum_cost_micro_usd"], "maximum_cost_micro_usd"
            ),
            authorization_digest=_strict_str(
                raw["authorization_digest"], "authorization_digest"
            ),
        )


@dataclass(frozen=True)
class PhysicalExecutionIdentityV1:
    schema_version: Literal[1]
    logical_attempt_id: str
    controller_id: str
    retry_ordinal: int
    physical_execution_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported physical execution identity schema")
        _require_digest(self.logical_attempt_id, "logical attempt id")
        _require_safe_id(self.controller_id, "controller id")
        _require_nonnegative_int(self.retry_ordinal, "retry ordinal")
        digest = stable_digest(self._unsigned())
        if self.physical_execution_id and self.physical_execution_id != digest:
            raise ValueError("physical execution identity digest does not match")
        object.__setattr__(self, "physical_execution_id", digest)

    @classmethod
    def create(
        cls, *, logical_attempt_id: str, controller_id: str, retry_ordinal: int
    ) -> PhysicalExecutionIdentityV1:
        return cls(1, logical_attempt_id, controller_id, retry_ordinal)

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_attempt_id": self.logical_attempt_id,
            "controller_id": self.controller_id,
            "retry_ordinal": self.retry_ordinal,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._unsigned(),
            "physical_execution_id": self.physical_execution_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PhysicalExecutionIdentityV1:
        _require_fields(
            raw,
            {
                "schema_version",
                "logical_attempt_id",
                "controller_id",
                "retry_ordinal",
                "physical_execution_id",
            },
            "physical execution identity",
        )
        return cls(
            schema_version=_strict_int(raw["schema_version"], "schema_version"),  # type: ignore[arg-type]
            logical_attempt_id=_strict_str(
                raw["logical_attempt_id"], "logical_attempt_id"
            ),
            controller_id=_strict_str(raw["controller_id"], "controller_id"),
            retry_ordinal=_strict_int(raw["retry_ordinal"], "retry_ordinal"),
            physical_execution_id=_strict_str(
                raw["physical_execution_id"], "physical_execution_id"
            ),
        )


@dataclass(frozen=True)
class ExecutionEventV1:
    schema_version: Literal[1]
    event_index: int
    previous_event_digest: str | None
    event_kind: EventKind
    controller_id: str
    schedule_digest: str
    logical_attempt_id: str | None
    physical_execution_id: str | None
    payload: dict[str, Any]
    recorded_at: str
    event_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported execution event schema")
        _require_nonnegative_int(self.event_index, "event index")
        if self.event_index == 0 and self.previous_event_digest is not None:
            raise ValueError("first execution event cannot have a predecessor")
        if self.event_index and self.previous_event_digest is None:
            raise ValueError("non-first execution event requires a predecessor")
        if self.previous_event_digest is not None:
            _require_digest(self.previous_event_digest, "previous event digest")
        if self.event_kind not in {
            "controller_initialized",
            "stage_authorized",
            "physical_execution_leased",
            "physical_execution_started",
            "physical_execution_terminal",
            "physical_execution_cleanup",
            "physical_execution_cost",
            "canonical_result_selected",
            "fatal_integrity_halt",
        }:
            raise ValueError("unknown execution event kind")
        _require_safe_id(self.controller_id, "controller id")
        _require_digest(self.schedule_digest, "schedule digest")
        if self.logical_attempt_id is not None:
            _require_digest(self.logical_attempt_id, "logical attempt id")
        if self.physical_execution_id is not None:
            _require_digest(self.physical_execution_id, "physical execution id")
        if not isinstance(self.payload, dict):
            raise ValueError("execution event payload must be an object")
        try:
            recorded = datetime.fromisoformat(self.recorded_at)
        except ValueError as exc:
            raise ValueError("execution event timestamp is invalid") from exc
        if recorded.tzinfo is None:
            raise ValueError("execution event timestamp must include a timezone")
        digest = stable_digest(self._unsigned())
        if self.event_digest and self.event_digest != digest:
            raise ValueError("execution event digest does not match")
        object.__setattr__(self, "event_digest", digest)

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_index": self.event_index,
            "previous_event_digest": self.previous_event_digest,
            "event_kind": self.event_kind,
            "controller_id": self.controller_id,
            "schedule_digest": self.schedule_digest,
            "logical_attempt_id": self.logical_attempt_id,
            "physical_execution_id": self.physical_execution_id,
            "payload": self.payload,
            "recorded_at": self.recorded_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "event_digest": self.event_digest}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExecutionEventV1:
        _require_fields(
            raw,
            {
                "schema_version",
                "event_index",
                "previous_event_digest",
                "event_kind",
                "controller_id",
                "schedule_digest",
                "logical_attempt_id",
                "physical_execution_id",
                "payload",
                "recorded_at",
                "event_digest",
            },
            "execution event",
        )
        return cls(
            schema_version=_strict_int(raw["schema_version"], "schema_version"),  # type: ignore[arg-type]
            event_index=_strict_int(raw["event_index"], "event_index"),
            previous_event_digest=(
                _strict_str(raw["previous_event_digest"], "previous_event_digest")
                if raw["previous_event_digest"] is not None
                else None
            ),
            event_kind=_strict_str(raw["event_kind"], "event_kind"),  # type: ignore[arg-type]
            controller_id=_strict_str(raw["controller_id"], "controller_id"),
            schedule_digest=_strict_str(raw["schedule_digest"], "schedule_digest"),
            logical_attempt_id=(
                _strict_str(raw["logical_attempt_id"], "logical_attempt_id")
                if raw["logical_attempt_id"] is not None
                else None
            ),
            physical_execution_id=(
                _strict_str(raw["physical_execution_id"], "physical_execution_id")
                if raw["physical_execution_id"] is not None
                else None
            ),
            payload=dict(_strict_mapping(raw["payload"], "payload")),
            recorded_at=_strict_str(raw["recorded_at"], "recorded_at"),
            event_digest=_strict_str(raw["event_digest"], "event_digest"),
        )


@dataclass
class PhysicalExecutionStateV1:
    identity: PhysicalExecutionIdentityV1
    reserve_micro_usd: int
    admission_weight: int
    planned_harbor_resources: tuple[str, ...]
    started: bool = False
    observed_harbor_resources: tuple[str, ...] = ()
    terminal_kind: str | None = None
    result_reference: str | None = None
    cell_outcome: dict[str, Any] | None = None
    cleanup_observed: bool = False
    cleanup_verified: bool = False
    cleanup_scope_verified: bool = False
    cleanup_post_run_inventory: bool = False
    discovered_harbor_resources: tuple[str, ...] = ()
    inspected_harbor_resources: tuple[str, ...] = ()
    removed_harbor_resources: tuple[str, ...] = ()
    remaining_harbor_resources: tuple[str, ...] = ()
    cleanup_receipt_reference: str | None = None
    cost_observed: bool = False
    cost_authoritative: bool = False
    actual_cost_micro_usd: int | None = None
    cost_source: str | None = None
    cost_receipt_reference: str | None = None
    cost_lease_exceeded: bool = False
    canonical: bool = False
    canonical_reconciliation: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self.terminal_kind is None

    @property
    def fully_reconciled(self) -> bool:
        return (
            self.terminal_kind is not None
            and self.cleanup_verified
            and self.cleanup_scope_verified
            and self.cleanup_post_run_inventory
            and self.cost_authoritative
            and self.actual_cost_micro_usd is not None
            and not self.cost_lease_exceeded
        )


@dataclass(frozen=True)
class ExecutionRecoverySnapshotV1:
    controller_id: str
    schedule_digest: str
    logical_attempt_count: int
    stage_authorizations: dict[str, StageExecutionAuthorizationV1]
    physical_executions: dict[str, PhysicalExecutionStateV1]
    canonical_results: dict[str, str]
    fatal_reasons: tuple[str, ...]
    event_count: int
    event_chain_digest: str

    @property
    def active_physical_execution_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                physical_id
                for physical_id, state in self.physical_executions.items()
                if state.active
            )
        )

    @property
    def complete(self) -> bool:
        return (
            len(self.canonical_results) == self.logical_attempt_count
            and not self.active_physical_execution_ids
            and not self.fatal_reasons
        )


@dataclass(frozen=True)
class AdmissionDecisionV1:
    status: Literal["admitted", "blocked", "complete"]
    physical_executions: tuple[PhysicalExecutionIdentityV1, ...]
    reason: str


@dataclass(frozen=True)
class CleanupObservationV1:
    verified: bool
    scope_verified: bool = False
    post_run_inventory: bool = False
    discovered_resources: tuple[str, ...] = ()
    inspected_resources: tuple[str, ...] = ()
    removed_resources: tuple[str, ...] = ()
    remaining_resources: tuple[str, ...] = ()
    receipt_reference: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.verified, bool):
            raise ValueError("cleanup verification must be boolean")
        if not isinstance(self.scope_verified, bool):
            raise ValueError("cleanup scope verification must be boolean")
        if not isinstance(self.post_run_inventory, bool):
            raise ValueError("cleanup inventory marker must be boolean")
        if self.verified and (not self.scope_verified or not self.post_run_inventory):
            raise ValueError(
                "verified cleanup requires exact scope and post-run inventory"
            )
        if self.verified and self.remaining_resources:
            raise ValueError("verified cleanup cannot retain Harbor resources")
        for values, label in (
            (self.discovered_resources, "discovered Harbor resource"),
            (self.inspected_resources, "inspected Harbor resource"),
            (self.removed_resources, "removed Harbor resource"),
            (self.remaining_resources, "remaining Harbor resource"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label}s must be unique")
            for value in values:
                _require_safe_id(value, label)
        inspected = set(self.inspected_resources)
        if not set(self.removed_resources).issubset(inspected):
            raise ValueError("removed Harbor resources were not inspected")
        if not set(self.remaining_resources).issubset(inspected):
            raise ValueError("remaining Harbor resources were not inspected")
        if not set(self.discovered_resources).issubset(inspected):
            raise ValueError("discovered Harbor resources were not inspected")
        if not set(self.discovered_resources).issubset(
            set(self.removed_resources) | set(self.remaining_resources)
        ):
            raise ValueError("every discovered Harbor resource must be reconciled")
        if self.receipt_reference is not None and not isinstance(
            self.receipt_reference, str
        ):
            raise ValueError("cleanup receipt reference must be a string")
        if self.receipt_reference is not None and not self.receipt_reference.strip():
            raise ValueError("cleanup receipt reference cannot be empty")
        if self.verified and self.receipt_reference is None:
            raise ValueError("verified cleanup requires a receipt reference")


@dataclass(frozen=True)
class CostObservationV1:
    actual_cost_micro_usd: int | None
    authoritative: bool
    source: str
    receipt_reference: str

    def __post_init__(self) -> None:
        if self.actual_cost_micro_usd is not None:
            _require_nonnegative_int(self.actual_cost_micro_usd, "actual cost")
        if not isinstance(self.authoritative, bool):
            raise ValueError("cost authority must be boolean")
        if self.authoritative and self.actual_cost_micro_usd is None:
            raise ValueError("authoritative cost cannot be missing")
        _require_safe_id(self.source, "cost source")
        if not self.receipt_reference.strip():
            raise ValueError("cost observation requires a receipt reference")


@dataclass(frozen=True)
class InterruptedExecutionObservationV1:
    """Authoritative observation for work left active by controller death."""

    terminal_kind: Literal[
        "success",
        "task_failure",
        "agent_timeout",
        "cancelled",
        "runner_start_failure",
        "sandbox_lost",
        "transport_interrupted",
    ]
    cell_outcome: CellOutcome
    cleanup: CleanupObservationV1
    cost: CostObservationV1
    reconciliation_receipt_reference: str
    reconciliation_receipt_sha256: str
    result_reference: str | None = None
    result_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.terminal_kind not in (_RETRYABLE_INFRASTRUCTURE | _BEHAVIORAL_TERMINAL):
            raise ValueError("interrupted observation has an unsupported terminal")
        if self.cell_outcome.terminal_kind != self.terminal_kind:
            raise ValueError("interrupted outcome terminal kind disagrees")
        if not self.reconciliation_receipt_reference.strip():
            raise ValueError("interrupted reconciliation requires a receipt")
        _require_digest(
            self.reconciliation_receipt_sha256,
            "interrupted reconciliation receipt digest",
        )
        if self.terminal_kind in _BEHAVIORAL_TERMINAL:
            if self.result_reference is None or not self.result_reference.strip():
                raise ValueError("completed interrupted work requires a durable result")
            if self.result_sha256 is None:
                raise ValueError("completed interrupted work requires a result digest")
            _require_digest(self.result_sha256, "interrupted result digest")
        elif self.result_reference is not None or self.result_sha256 is not None:
            raise ValueError(
                "infrastructure interruption cannot claim a behavioral result"
            )


@dataclass(frozen=True)
class CanonicalizationObservationV1:
    """Adapter-owned proof that required downstream evidence is finalized."""

    kind: str
    status: Literal["verified"]
    reference: str
    evidence_digest: str

    def __post_init__(self) -> None:
        _require_safe_id(self.kind, "canonicalization kind")
        if self.status != "verified":
            raise ValueError("canonicalization observation must be verified")
        if not self.reference.strip():
            raise ValueError("canonicalization requires an evidence reference")
        _require_digest(self.evidence_digest, "canonical evidence digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "reference": self.reference,
            "evidence_digest": self.evidence_digest,
        }


class ExecutionRecoveryStore:
    """Append-only, fsynced event journal for one frozen schedule."""

    def __init__(
        self,
        path: Path,
        *,
        controller_id: str,
        schedule: ExecutionScheduleV1,
        redaction_secrets: Sequence[str] = (),
    ) -> None:
        self.path = path
        self.controller_id = controller_id
        self.schedule = schedule
        self.redaction_secrets = tuple(
            sorted(
                set(redaction_secrets) | set(secrets_from_env(os.environ)),
                key=len,
                reverse=True,
            )
        )
        _require_safe_id(controller_id, "controller id")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(f"{path}.lock")
        self._events_cache: list[ExecutionEventV1] | None = None
        self._cache_signature: tuple[int, int, int, int] | None = None
        with self._lock:
            events = self._read_unlocked()
            if not events:
                self._append_unlocked(
                    events,
                    "controller_initialized",
                    payload={"schedule": schedule.to_dict()},
                )
            else:
                first = events[0]
                if (
                    first.event_kind != "controller_initialized"
                    or first.payload.get("schedule") != schedule.to_dict()
                ):
                    raise ExecutionJournalCorrupt(
                        "execution journal is bound to a different schedule"
                    )

    def snapshot(self) -> ExecutionRecoverySnapshotV1:
        with self._lock:
            return self._snapshot(self._read_unlocked())

    def activate_redaction_secrets(self, secrets: Sequence[str]) -> None:
        additions = {value for value in secrets if value}
        if not additions.issubset(self.redaction_secrets):
            with self._lock:
                events = self._read_unlocked()
                pending = tuple(additions - set(self.redaction_secrets))
                if any(
                    _contains_sensitive_text(event.payload, pending) for event in events
                ):
                    raise ExecutionRecoveryError(
                        "existing execution journal contains sensitive data"
                    )
                self.redaction_secrets = tuple(
                    sorted(
                        set(self.redaction_secrets) | additions,
                        key=len,
                        reverse=True,
                    )
                )

    def _append_unlocked(
        self,
        events: list[ExecutionEventV1],
        event_kind: EventKind,
        *,
        logical_attempt_id: str | None = None,
        physical_execution_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> ExecutionEventV1:
        raw_payload = dict(payload or {})
        if _contains_sensitive_text(raw_payload, self.redaction_secrets):
            raise ExecutionRecoveryError(
                "durable execution event payload contains sensitive data"
            )
        event = ExecutionEventV1(
            schema_version=1,
            event_index=len(events),
            previous_event_digest=(events[-1].event_digest if events else None),
            event_kind=event_kind,
            controller_id=self.controller_id,
            schedule_digest=self.schedule.schedule_digest,
            logical_attempt_id=logical_attempt_id,
            physical_execution_id=physical_execution_id,
            payload=raw_payload,
            recorded_at=datetime.now(UTC).isoformat(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if created:
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        self._events_cache = [*events, event]
        self._cache_signature = self._path_signature()
        return event

    def _read_unlocked(self) -> list[ExecutionEventV1]:
        if not self.path.exists():
            return []
        signature = self._path_signature()
        if self._events_cache is not None and self._cache_signature == signature:
            return list(self._events_cache)
        events: list[ExecutionEventV1] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                event = ExecutionEventV1.from_dict(
                    _strict_mapping(raw, "execution event")
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ExecutionJournalCorrupt(
                    f"execution journal row {line_number} is invalid"
                ) from exc
            if event.event_index != len(events):
                raise ExecutionJournalCorrupt(
                    "execution journal event indexes are not contiguous"
                )
            expected_previous = events[-1].event_digest if events else None
            if event.previous_event_digest != expected_previous:
                raise ExecutionJournalCorrupt(
                    "execution journal digest chain is broken"
                )
            if (
                event.controller_id != self.controller_id
                or event.schedule_digest != self.schedule.schedule_digest
            ):
                raise ExecutionJournalCorrupt("execution journal identity changed")
            events.append(event)
        self._events_cache = list(events)
        self._cache_signature = signature
        return events

    def _path_signature(self) -> tuple[int, int, int, int]:
        stat = self.path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _snapshot(  # noqa: C901 - validates every durable state transition
        self, events: Sequence[ExecutionEventV1]
    ) -> ExecutionRecoverySnapshotV1:
        physical: dict[str, PhysicalExecutionStateV1] = {}
        canonical: dict[str, str] = {}
        authorizations: dict[str, StageExecutionAuthorizationV1] = {}
        fatal: list[str] = []
        plans = {
            item.logical_attempt_id: item for item in self.schedule.logical_attempts
        }
        for event in events:
            try:
                _require_fields(
                    event.payload,
                    _EVENT_PAYLOAD_FIELDS[event.event_kind],
                    f"{event.event_kind} payload",
                )
            except (KeyError, ValueError) as exc:
                raise ExecutionJournalCorrupt(
                    f"{event.event_kind} payload is invalid"
                ) from exc
            logical_id = event.logical_attempt_id
            physical_id = event.physical_execution_id
            if event.event_kind == "controller_initialized":
                continue
            if event.event_kind == "stage_authorized":
                if logical_id is not None or physical_id is not None:
                    raise ExecutionJournalCorrupt(
                        "stage authorization cannot carry a physical identity"
                    )
                authorization = StageExecutionAuthorizationV1.from_dict(
                    _strict_mapping(event.payload.get("authorization"), "authorization")
                )
                if authorization.schedule_digest != self.schedule.schedule_digest:
                    raise ExecutionJournalCorrupt(
                        "stage authorization targets a different schedule"
                    )
                prior = authorizations.get(authorization.stage_id)
                if prior is not None and prior != authorization:
                    raise ExecutionJournalCorrupt(
                        "stage authorization changed after it was recorded"
                    )
                authorizations[authorization.stage_id] = authorization
                continue
            if event.event_kind == "fatal_integrity_halt":
                reason = event.payload.get("reason")
                if not isinstance(reason, str) or not reason:
                    raise ExecutionJournalCorrupt("fatal event has no reason")
                fatal.append(reason)
                continue
            if logical_id is None or physical_id is None:
                raise ExecutionJournalCorrupt(
                    "physical execution event is missing identity"
                )
            if event.event_kind == "physical_execution_leased":
                if physical_id in physical:
                    raise ExecutionJournalCorrupt(
                        "physical execution was leased more than once"
                    )
                identity = PhysicalExecutionIdentityV1.from_dict(
                    _strict_mapping(event.payload.get("identity"), "identity")
                )
                if (
                    identity.physical_execution_id != physical_id
                    or identity.logical_attempt_id != logical_id
                    or identity.controller_id != self.controller_id
                ):
                    raise ExecutionJournalCorrupt(
                        "physical lease identity does not match its event"
                    )
                resources = tuple(
                    _strict_str(item, "planned_harbor_resource")
                    for item in _strict_list(
                        event.payload.get("planned_harbor_resources"),
                        "planned_harbor_resources",
                    )
                )
                plan = plans.get(logical_id)
                if plan is None:
                    raise ExecutionJournalCorrupt(
                        "physical lease is outside the frozen schedule"
                    )
                reserve = _strict_int(
                    event.payload.get("reserve_micro_usd"),
                    "reserve_micro_usd",
                )
                weight = _strict_int(
                    event.payload.get("admission_weight"),
                    "admission_weight",
                )
                expected_retry = sum(
                    state.identity.logical_attempt_id == logical_id
                    for state in physical.values()
                )
                if (
                    reserve != plan.maximum_cost_micro_usd
                    or weight != plan.admission_weight
                    or resources != plan.planned_harbor_resources
                    or identity.retry_ordinal != expected_retry
                ):
                    raise ExecutionJournalCorrupt(
                        "physical lease does not match the frozen logical plan"
                    )
                physical[physical_id] = PhysicalExecutionStateV1(
                    identity=identity,
                    reserve_micro_usd=reserve,
                    admission_weight=weight,
                    planned_harbor_resources=resources,
                )
                continue
            state = physical.get(physical_id)
            if state is None or state.identity.logical_attempt_id != logical_id:
                raise ExecutionJournalCorrupt(
                    "execution event refers to an unknown physical identity"
                )
            if event.event_kind == "physical_execution_started":
                if state.started or state.terminal_kind is not None:
                    raise ExecutionJournalCorrupt(
                        "physical execution start transition is invalid"
                    )
                resources = tuple(
                    _strict_str(item, "observed_harbor_resource")
                    for item in _strict_list(
                        event.payload.get("observed_harbor_resources"),
                        "observed_harbor_resources",
                    )
                )
                if len(set(resources)) != len(resources) or not set(
                    state.planned_harbor_resources
                ).issubset(resources):
                    raise ExecutionJournalCorrupt(
                        "observed Harbor resources omit a planned resource"
                    )
                state.started = True
                state.observed_harbor_resources = resources
            elif event.event_kind == "physical_execution_terminal":
                terminal_kind = event.payload.get("terminal_kind")
                if (
                    not isinstance(terminal_kind, str)
                    or terminal_kind not in _TERMINAL_KINDS
                    or state.terminal_kind is not None
                ):
                    raise ExecutionJournalCorrupt(
                        "physical terminal transition is invalid"
                    )
                state.terminal_kind = terminal_kind
                result_reference = event.payload.get("result_reference")
                if result_reference is not None and not isinstance(
                    result_reference, str
                ):
                    raise ExecutionJournalCorrupt(
                        "physical result reference must be a string"
                    )
                state.result_reference = result_reference
                raw_outcome = event.payload.get("cell_outcome")
                try:
                    outcome_record = dict(_strict_mapping(raw_outcome, "cell_outcome"))
                    parsed_outcome = _cell_outcome_from_record(outcome_record)
                except (TypeError, ValueError) as exc:
                    raise ExecutionJournalCorrupt(
                        "physical terminal outcome is invalid"
                    ) from exc
                expected_status = (
                    "passed"
                    if terminal_kind == "success"
                    else "cancelled"
                    if terminal_kind == "cancelled"
                    else "failed"
                )
                if (
                    parsed_outcome.status != expected_status
                    or parsed_outcome.terminal_kind != terminal_kind
                ):
                    raise ExecutionJournalCorrupt(
                        "physical terminal kind disagrees with its outcome"
                    )
                if terminal_kind in _BEHAVIORAL_TERMINAL and (
                    not isinstance(result_reference, str)
                    or not result_reference.strip()
                ):
                    raise ExecutionJournalCorrupt(
                        "behavioral terminal lacks a result reference"
                    )
                if (
                    terminal_kind
                    in {
                        "success",
                        "task_failure",
                        "agent_timeout",
                    }
                    and not state.started
                ):
                    raise ExecutionJournalCorrupt(
                        "behavioral terminal was recorded before execution start"
                    )
                if (
                    terminal_kind
                    in {
                        "sandbox_lost",
                        "transport_interrupted",
                    }
                    and not state.started
                ):
                    raise ExecutionJournalCorrupt(
                        "post-start infrastructure terminal lacks a start event"
                    )
                state.cell_outcome = outcome_record
            elif event.event_kind == "physical_execution_cleanup":
                if state.terminal_kind is None or state.cleanup_observed:
                    raise ExecutionJournalCorrupt(
                        "physical cleanup transition is invalid"
                    )
                remaining = tuple(
                    _strict_str(item, "remaining_harbor_resource")
                    for item in _strict_list(
                        event.payload.get("remaining_resources"),
                        "remaining_resources",
                    )
                )
                verified = event.payload.get("verified")
                if not isinstance(verified, bool):
                    raise ExecutionJournalCorrupt(
                        "physical cleanup verification must be boolean"
                    )
                scope_verified = event.payload.get("scope_verified")
                post_run_inventory = event.payload.get("post_run_inventory")
                if not isinstance(scope_verified, bool) or not isinstance(
                    post_run_inventory, bool
                ):
                    raise ExecutionJournalCorrupt(
                        "cleanup scope and inventory markers must be boolean"
                    )
                discovered = tuple(
                    _strict_str(item, "discovered_harbor_resource")
                    for item in _strict_list(
                        event.payload.get("discovered_resources"),
                        "discovered_resources",
                    )
                )
                inspected = tuple(
                    _strict_str(item, "inspected_harbor_resource")
                    for item in _strict_list(
                        event.payload.get("inspected_resources"),
                        "inspected_resources",
                    )
                )
                removed = tuple(
                    _strict_str(item, "removed_harbor_resource")
                    for item in _strict_list(
                        event.payload.get("removed_resources"),
                        "removed_resources",
                    )
                )
                if any(
                    len(set(values)) != len(values)
                    for values in (discovered, inspected, removed, remaining)
                ):
                    raise ExecutionJournalCorrupt(
                        "cleanup resource identities must be unique"
                    )
                expected = (
                    set(state.planned_harbor_resources)
                    | set(state.observed_harbor_resources)
                    | set(discovered)
                )
                if (
                    not expected.issubset(inspected)
                    or not set(removed).issubset(inspected)
                    or not set(remaining).issubset(inspected)
                    or not set(discovered).issubset(set(removed) | set(remaining))
                    or (verified and remaining)
                    or (verified and (not scope_verified or not post_run_inventory))
                ):
                    raise ExecutionJournalCorrupt(
                        "cleanup did not reconcile exact Harbor resources"
                    )
                state.cleanup_observed = True
                state.cleanup_verified = verified and not remaining
                state.cleanup_scope_verified = scope_verified
                state.cleanup_post_run_inventory = post_run_inventory
                state.discovered_harbor_resources = discovered
                state.inspected_harbor_resources = inspected
                state.removed_harbor_resources = removed
                state.remaining_harbor_resources = remaining
                receipt_reference = event.payload.get("receipt_reference")
                if receipt_reference is not None and not isinstance(
                    receipt_reference, str
                ):
                    raise ExecutionJournalCorrupt(
                        "cleanup receipt reference must be a string"
                    )
                if verified and (
                    not isinstance(receipt_reference, str)
                    or not receipt_reference.strip()
                ):
                    raise ExecutionJournalCorrupt(
                        "verified cleanup lacks a receipt reference"
                    )
                state.cleanup_receipt_reference = receipt_reference
            elif event.event_kind == "physical_execution_cost":
                if state.terminal_kind is None or state.cost_authoritative:
                    raise ExecutionJournalCorrupt("physical cost transition is invalid")
                authoritative = event.payload.get("authoritative")
                if not isinstance(authoritative, bool):
                    raise ExecutionJournalCorrupt(
                        "physical cost authority must be boolean"
                    )
                raw_cost = event.payload.get("actual_cost_micro_usd")
                if raw_cost is not None:
                    raw_cost = _strict_int(raw_cost, "actual_cost_micro_usd")
                    _require_nonnegative_int(raw_cost, "actual cost")
                if authoritative and raw_cost is None:
                    raise ExecutionJournalCorrupt(
                        "authoritative physical cost cannot be missing"
                    )
                lease_exceeded = event.payload.get("lease_exceeded")
                if not isinstance(lease_exceeded, bool) or lease_exceeded != (
                    raw_cost is not None and raw_cost > state.reserve_micro_usd
                ):
                    raise ExecutionJournalCorrupt(
                        "physical cost lease status is invalid"
                    )
                state.cost_observed = True
                state.cost_authoritative = authoritative
                state.actual_cost_micro_usd = raw_cost
                state.cost_lease_exceeded = lease_exceeded
                source = event.payload.get("source")
                receipt = event.payload.get("receipt_reference")
                if not isinstance(source, str) or not source:
                    raise ExecutionJournalCorrupt("physical cost source is invalid")
                if not isinstance(receipt, str) or not receipt.strip():
                    raise ExecutionJournalCorrupt("physical cost receipt is invalid")
                state.cost_source = source
                state.cost_receipt_reference = receipt
            elif event.event_kind == "canonical_result_selected":
                if logical_id in canonical or state.canonical:
                    raise ExecutionJournalCorrupt(
                        "logical result was selected more than once"
                    )
                if not state.fully_reconciled:
                    raise ExecutionJournalCorrupt(
                        "canonical result is not fully reconciled"
                    )
                if state.terminal_kind not in _BEHAVIORAL_TERMINAL:
                    raise ExecutionJournalCorrupt(
                        "infrastructure or integrity failure cannot be canonical"
                    )
                if state.cell_outcome is None or not state.result_reference:
                    raise ExecutionJournalCorrupt(
                        "canonical result lacks normalized terminal evidence"
                    )
                if event.payload.get("result_reference") != state.result_reference:
                    raise ExecutionJournalCorrupt("canonical result reference changed")
                try:
                    reconciliation = CanonicalizationObservationV1(
                        **dict(
                            _strict_mapping(
                                event.payload.get("reconciliation"),
                                "canonical reconciliation",
                            )
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise ExecutionJournalCorrupt(
                        "canonical reconciliation is invalid"
                    ) from exc
                state.canonical = True
                state.canonical_reconciliation = reconciliation.to_dict()
                canonical[logical_id] = physical_id
        chain = events[-1].event_digest if events else stable_digest([])
        return ExecutionRecoverySnapshotV1(
            controller_id=self.controller_id,
            schedule_digest=self.schedule.schedule_digest,
            logical_attempt_count=len(self.schedule.logical_attempts),
            stage_authorizations=authorizations,
            physical_executions=physical,
            canonical_results=canonical,
            fatal_reasons=tuple(fatal),
            event_count=len(events),
            event_chain_digest=chain,
        )


class ExecutionRecoveryController:
    """Durable weighted scheduler around Fugue's canonical cell executor."""

    def __init__(
        self,
        path: Path,
        *,
        controller_id: str,
        schedule: ExecutionScheduleV1,
        redaction_secrets: Sequence[str] = (),
    ) -> None:
        self.schedule = schedule
        self.controller_id = controller_id
        self.store = ExecutionRecoveryStore(
            path,
            controller_id=controller_id,
            schedule=schedule,
            redaction_secrets=redaction_secrets,
        )
        self._plan_by_id = {
            item.logical_attempt_id: item for item in schedule.logical_attempts
        }

    def snapshot(self) -> ExecutionRecoverySnapshotV1:
        return self.store.snapshot()

    def activate_redaction_secrets(self, secrets: Sequence[str]) -> None:
        """Protect all later journal events with the complete execution set."""

        self.store.activate_redaction_secrets(secrets)

    def authorize_stage(self, authorization: StageExecutionAuthorizationV1) -> None:
        if authorization.schedule_digest != self.schedule.schedule_digest:
            raise ValueError("stage authorization targets a different schedule")
        plans = [
            item
            for item in self.schedule.logical_attempts
            if item.stage_id == authorization.stage_id
        ]
        if not plans:
            raise ValueError("stage authorization names an unknown stage")
        if len(plans) != authorization.maximum_logical_attempts:
            raise ValueError("stage authorization logical ceiling is not exact")
        planned_cost = sum(item.maximum_cost_micro_usd for item in plans)
        if planned_cost > authorization.maximum_cost_micro_usd:
            raise ValueError("stage authorization cost ceiling is below its plan")
        maximum_possible = (
            len(plans) + self.schedule.maximum_infrastructure_replacements
        )
        if authorization.maximum_physical_executions > maximum_possible:
            raise ValueError("stage authorization physical ceiling exceeds policy")
        with self.store._lock:
            events = self.store._read_unlocked()
            snapshot = self.store._snapshot(events)
            if any(
                prior.preview_digest != authorization.preview_digest
                for prior in snapshot.stage_authorizations.values()
            ):
                raise ExecutionRecoveryError(
                    "stage authorization targets a different full preview"
                )
            prior = snapshot.stage_authorizations.get(authorization.stage_id)
            if prior is not None:
                if prior != authorization:
                    raise ExecutionRecoveryError(
                        "stage already has a different durable authorization"
                    )
                return
            event = self.store._append_unlocked(
                events,
                "stage_authorized",
                payload={"authorization": authorization.to_dict()},
            )
            events.append(event)

    def admit(  # noqa: C901 - one atomic admission transaction
        self, *, stage_id: str
    ) -> AdmissionDecisionV1:
        """Atomically lease the next capacity-bounded slice of one block."""

        with self.store._lock:  # one check-and-append transaction
            events = self.store._read_unlocked()
            snapshot = self.store._snapshot(events)
            authorization = snapshot.stage_authorizations.get(stage_id)
            if authorization is None:
                return AdmissionDecisionV1(
                    "blocked", (), "stage lacks a durable authorization"
                )
            if snapshot.fatal_reasons:
                return AdmissionDecisionV1(
                    "blocked", (), f"fatal integrity halt: {snapshot.fatal_reasons[-1]}"
                )
            if any(
                state.cost_lease_exceeded
                for state in snapshot.physical_executions.values()
            ):
                return AdmissionDecisionV1(
                    "blocked", (), "a physical execution exceeded its cost lease"
                )
            if any(
                state.terminal_kind is not None and not state.cost_authoritative
                for state in snapshot.physical_executions.values()
            ):
                return AdmissionDecisionV1(
                    "blocked", (), "authoritative cost is missing"
                )
            if any(
                state.terminal_kind is not None and not state.cleanup_verified
                for state in snapshot.physical_executions.values()
            ):
                return AdmissionDecisionV1(
                    "blocked", (), "terminal cleanup is not verified"
                )
            active = [
                state for state in snapshot.physical_executions.values() if state.active
            ]
            active_weight = sum(item.admission_weight for item in active)
            active_cost = sum(item.reserve_micro_usd for item in active)
            settled_cost = sum(
                item.actual_cost_micro_usd or 0
                for item in snapshot.physical_executions.values()
                if item.cost_authoritative
            )
            stage_plans = [
                item
                for item in self.schedule.logical_attempts
                if item.stage_id == stage_id
            ]
            stage_logical_ids = {item.logical_attempt_id for item in stage_plans}
            stage_executions = [
                item
                for item in snapshot.physical_executions.values()
                if item.identity.logical_attempt_id in stage_logical_ids
            ]
            stage_settled_cost = sum(
                item.actual_cost_micro_usd or 0
                for item in stage_executions
                if item.cost_authoritative
            )
            stage_active_cost = sum(
                item.reserve_micro_usd for item in stage_executions if item.active
            )
            if len(snapshot.physical_executions) >= (
                self.schedule.maximum_physical_executions
            ):
                if len(snapshot.canonical_results) == len(
                    self.schedule.logical_attempts
                ):
                    return AdmissionDecisionV1("complete", (), "matrix complete")
                return AdmissionDecisionV1(
                    "blocked", (), "physical execution ceiling reached"
                )
            pending_by_block: dict[tuple[int, int], list[LogicalExecutionPlanV1]] = {}
            for plan in stage_plans:
                if plan.logical_attempt_id in snapshot.canonical_results:
                    continue
                executions = [
                    state
                    for state in snapshot.physical_executions.values()
                    if state.identity.logical_attempt_id == plan.logical_attempt_id
                ]
                if any(state.active for state in executions):
                    continue
                if executions:
                    latest = max(
                        executions, key=lambda item: item.identity.retry_ordinal
                    )
                    if latest.terminal_kind in self.schedule.fatal_terminal_kinds:
                        return AdmissionDecisionV1(
                            "blocked",
                            (),
                            f"fatal integrity terminal: {latest.terminal_kind}",
                        )
                    if (
                        latest.terminal_kind
                        not in self.schedule.retryable_terminal_kinds
                    ):
                        raise ExecutionRecoveryError(
                            "terminal behavioral execution lacks canonical selection"
                        )
                    replacements = len(snapshot.physical_executions) - len(
                        {
                            state.identity.logical_attempt_id
                            for state in snapshot.physical_executions.values()
                        }
                    )
                    if replacements >= (
                        self.schedule.maximum_infrastructure_replacements
                    ):
                        return AdmissionDecisionV1(
                            "blocked",
                            (),
                            "infrastructure replacement ceiling reached",
                        )
                pending_by_block.setdefault(
                    (plan.stage_ordinal, plan.block_ordinal), []
                ).append(plan)
            if not pending_by_block:
                stage_active = [
                    item
                    for item in active
                    if item.identity.logical_attempt_id in stage_logical_ids
                ]
                if stage_active:
                    return AdmissionDecisionV1(
                        "blocked", (), "physical executions are still active"
                    )
                return AdmissionDecisionV1("complete", (), "authorized stage complete")
            block_key = min(pending_by_block)
            earlier_unreconciled = [
                plan
                for plan in self.schedule.logical_attempts
                if (plan.stage_ordinal, plan.block_ordinal) < block_key
                and plan.logical_attempt_id not in snapshot.canonical_results
            ]
            if earlier_unreconciled:
                return AdmissionDecisionV1(
                    "blocked", (), "an earlier admission block is incomplete"
                )
            capacity_left = self.schedule.capacity_units - active_weight
            execution_slots = self.schedule.maximum_in_flight_executions - len(active)
            physical_slots = self.schedule.maximum_physical_executions - len(
                snapshot.physical_executions
            )
            replacement_count = len(snapshot.physical_executions) - len(
                {
                    state.identity.logical_attempt_id
                    for state in snapshot.physical_executions.values()
                }
            )
            replacement_slots = (
                self.schedule.maximum_infrastructure_replacements - replacement_count
            )
            in_flight_cost_left = (
                self.schedule.maximum_in_flight_micro_usd - active_cost
            )
            total_cost_left = (
                self.schedule.maximum_total_micro_usd - settled_cost - active_cost
            )
            stage_cost_left = (
                authorization.maximum_cost_micro_usd
                - stage_settled_cost
                - stage_active_cost
            )
            stage_physical_left = authorization.maximum_physical_executions - len(
                stage_executions
            )
            selected: list[LogicalExecutionPlanV1] = []
            for plan in pending_by_block[block_key]:
                if len(selected) >= min(
                    execution_slots, physical_slots, stage_physical_left
                ):
                    break
                prior_count = sum(
                    state.identity.logical_attempt_id == plan.logical_attempt_id
                    for state in snapshot.physical_executions.values()
                )
                if prior_count and replacement_slots < 1:
                    continue
                if plan.admission_weight > capacity_left:
                    continue
                if plan.maximum_cost_micro_usd > min(
                    in_flight_cost_left, total_cost_left, stage_cost_left
                ):
                    continue
                selected.append(plan)
                capacity_left -= plan.admission_weight
                in_flight_cost_left -= plan.maximum_cost_micro_usd
                total_cost_left -= plan.maximum_cost_micro_usd
                stage_cost_left -= plan.maximum_cost_micro_usd
                if prior_count:
                    replacement_slots -= 1
            if not selected:
                return AdmissionDecisionV1(
                    "blocked", (), "capacity or budget prevents admission"
                )
            identities: list[PhysicalExecutionIdentityV1] = []
            for plan in selected:
                retry_ordinal = sum(
                    state.identity.logical_attempt_id == plan.logical_attempt_id
                    for state in snapshot.physical_executions.values()
                )
                identity = PhysicalExecutionIdentityV1.create(
                    logical_attempt_id=plan.logical_attempt_id,
                    controller_id=self.controller_id,
                    retry_ordinal=retry_ordinal,
                )
                event = self.store._append_unlocked(
                    events,
                    "physical_execution_leased",
                    logical_attempt_id=plan.logical_attempt_id,
                    physical_execution_id=identity.physical_execution_id,
                    payload={
                        "identity": identity.to_dict(),
                        "reserve_micro_usd": plan.maximum_cost_micro_usd,
                        "admission_weight": plan.admission_weight,
                        "planned_harbor_resources": list(plan.planned_harbor_resources),
                    },
                )
                events.append(event)
                identities.append(identity)
            return AdmissionDecisionV1("admitted", tuple(identities), "admitted")

    def started(
        self,
        physical: PhysicalExecutionIdentityV1,
        *,
        observed_harbor_resources: Sequence[str],
    ) -> None:
        observed = tuple(observed_harbor_resources)
        if len(set(observed)) != len(observed):
            raise ValueError("observed Harbor resources must be unique")
        for resource in observed:
            _require_safe_id(resource, "observed Harbor resource")
        with self.store._lock:
            events, _snapshot, state = self._locked_state(physical)
            if state.started or state.terminal_kind is not None:
                raise ExecutionRecoveryError(
                    "physical execution cannot transition to started"
                )
            if not set(state.planned_harbor_resources).issubset(observed):
                raise ExecutionRecoveryError(
                    "observed Harbor resources omit a planned resource"
                )
            self._append_locked(
                events,
                "physical_execution_started",
                physical,
                {"observed_harbor_resources": list(observed)},
            )

    def terminal(
        self,
        physical: PhysicalExecutionIdentityV1,
        *,
        terminal_kind: TerminalKind,
        result_reference: str | None,
        cell_outcome: CellOutcome,
    ) -> None:
        payload = self._terminal_payload(
            terminal_kind=terminal_kind,
            result_reference=result_reference,
            cell_outcome=cell_outcome,
        )
        with self.store._lock:
            events, _snapshot, state = self._locked_state(physical)
            self._validate_terminal_transition(state, terminal_kind)
            self._append_locked(
                events, "physical_execution_terminal", physical, payload
            )
            if terminal_kind in self.schedule.fatal_terminal_kinds:
                self._append_halt_locked(
                    events,
                    f"{physical.logical_attempt_id}: {terminal_kind}",
                    physical,
                )

    def cleanup(
        self,
        physical: PhysicalExecutionIdentityV1,
        observation: CleanupObservationV1,
    ) -> None:
        with self.store._lock:
            events, _snapshot, state = self._locked_state(physical)
            self._validate_cleanup_transition(state, observation)
            self._append_locked(
                events,
                "physical_execution_cleanup",
                physical,
                self._cleanup_payload(observation),
            )
            if not observation.verified or observation.remaining_resources:
                self._append_halt_locked(
                    events,
                    f"{physical.logical_attempt_id}: cleanup_failure",
                    physical,
                )

    def cost(
        self,
        physical: PhysicalExecutionIdentityV1,
        observation: CostObservationV1,
    ) -> None:
        with self.store._lock:
            events, _snapshot, state = self._locked_state(physical)
            self._validate_cost_transition(
                state,
                actual_cost_micro_usd=observation.actual_cost_micro_usd,
                authoritative=observation.authoritative,
            )
            lease_exceeded = (
                observation.actual_cost_micro_usd is not None
                and observation.actual_cost_micro_usd > state.reserve_micro_usd
            )
            self._append_locked(
                events,
                "physical_execution_cost",
                physical,
                {
                    "actual_cost_micro_usd": observation.actual_cost_micro_usd,
                    "authoritative": observation.authoritative,
                    "lease_exceeded": lease_exceeded,
                    "source": observation.source,
                    "receipt_reference": observation.receipt_reference,
                },
            )
            if lease_exceeded:
                self._append_halt_locked(
                    events,
                    f"{physical.logical_attempt_id}: cost exceeded its lease",
                    physical,
                )

    def select_canonical(
        self,
        physical: PhysicalExecutionIdentityV1,
        reconciliation: CanonicalizationObservationV1,
    ) -> None:
        with self.store._lock:
            events, snapshot, state = self._locked_state(physical)
            self._validate_canonical_transition(snapshot, state, physical)
            self._append_locked(
                events,
                "canonical_result_selected",
                physical,
                {
                    "result_reference": state.result_reference,
                    "reconciliation": reconciliation.to_dict(),
                },
            )

    def finalize(
        self,
        physical: PhysicalExecutionIdentityV1,
        *,
        terminal_kind: TerminalKind,
        result_reference: str | None,
        cleanup: CleanupObservationV1,
        cost: CostObservationV1,
        cell_outcome: CellOutcome,
    ) -> None:
        terminal_payload = self._terminal_payload(
            terminal_kind=terminal_kind,
            result_reference=result_reference,
            cell_outcome=cell_outcome,
        )
        with self.store._lock:
            events, snapshot, state = self._locked_state(physical)
            self._validate_terminal_transition(state, terminal_kind)
            self._validate_cleanup_resources(state, cleanup)
            lease_exceeded = (
                cost.actual_cost_micro_usd is not None
                and cost.actual_cost_micro_usd > state.reserve_micro_usd
            )
            self._append_locked(
                events, "physical_execution_terminal", physical, terminal_payload
            )
            self._append_locked(
                events,
                "physical_execution_cleanup",
                physical,
                self._cleanup_payload(cleanup),
            )
            self._append_locked(
                events,
                "physical_execution_cost",
                physical,
                {
                    "actual_cost_micro_usd": cost.actual_cost_micro_usd,
                    "authoritative": cost.authoritative,
                    "lease_exceeded": lease_exceeded,
                    "source": cost.source,
                    "receipt_reference": cost.receipt_reference,
                },
            )
            fatal_reason = None
            if lease_exceeded:
                fatal_reason = f"{physical.logical_attempt_id}: cost exceeded its lease"
            elif terminal_kind in self.schedule.fatal_terminal_kinds:
                fatal_reason = f"{physical.logical_attempt_id}: {terminal_kind}"
            elif not cleanup.verified or cleanup.remaining_resources:
                fatal_reason = f"{physical.logical_attempt_id}: cleanup_failure"
            if fatal_reason is not None:
                self._append_halt_locked(events, fatal_reason, physical)
                return
            # Canonicalization is deliberately separate. A digest-bound adapter
            # must first prove that every required downstream evidence surface
            # has finalized; terminal+cleanup+cost alone is insufficient.

    def reconcile_interrupted_physical(
        self,
        physical: PhysicalExecutionIdentityV1,
        *,
        terminal_kind: Literal[
            "success",
            "task_failure",
            "agent_timeout",
            "cancelled",
            "runner_start_failure",
            "sandbox_lost",
            "transport_interrupted",
        ],
        cell_outcome: CellOutcome,
        result_reference: str | None,
        cleanup: CleanupObservationV1,
        cost: CostObservationV1,
    ) -> None:
        """Reconcile active work from external, authoritative evidence.

        Rehydration never guesses whether paid work completed. An operator or
        runner adapter must supply a typed terminal, exact cleanup receipt,
        normalized outcome, and authoritative cost. Completed behavioral work
        remains the same physical execution; only typed infrastructure loss is
        eligible for replacement.
        """

        if terminal_kind not in (
            set(self.schedule.retryable_terminal_kinds) | _BEHAVIORAL_TERMINAL
        ):
            raise ValueError("interrupted execution terminal is not allowed")
        self.finalize(
            physical,
            terminal_kind=terminal_kind,
            result_reference=result_reference,
            cleanup=cleanup,
            cost=cost,
            cell_outcome=cell_outcome,
        )

    def halt(
        self,
        reason: str,
        *,
        logical_attempt_id: str | None = None,
        physical_execution_id: str | None = None,
    ) -> None:
        if not reason.strip():
            raise ValueError("fatal integrity reason cannot be empty")
        with self.store._lock:
            events = self.store._read_unlocked()
            self.store._snapshot(events)
            event = self.store._append_unlocked(
                events,
                "fatal_integrity_halt",
                logical_attempt_id=logical_attempt_id,
                physical_execution_id=physical_execution_id,
                payload={"reason": redact_text(reason, self.store.redaction_secrets)},
            )
            events.append(event)

    def _locked_state(
        self,
        physical: PhysicalExecutionIdentityV1,
    ) -> tuple[
        list[ExecutionEventV1],
        ExecutionRecoverySnapshotV1,
        PhysicalExecutionStateV1,
    ]:
        if physical.controller_id != self.controller_id:
            raise ValueError("physical execution belongs to another controller")
        if physical.logical_attempt_id not in self._plan_by_id:
            raise ValueError("physical execution is outside the schedule")
        events = self.store._read_unlocked()
        snapshot = self.store._snapshot(events)
        if physical.physical_execution_id not in snapshot.physical_executions:
            raise ValueError("physical execution has not been leased")
        state = snapshot.physical_executions[physical.physical_execution_id]
        if state.identity != physical:
            raise ValueError("physical execution identity does not match its lease")
        return events, snapshot, state

    def _append_locked(
        self,
        events: list[ExecutionEventV1],
        event_kind: EventKind,
        physical: PhysicalExecutionIdentityV1,
        payload: Mapping[str, Any],
    ) -> None:
        event = self.store._append_unlocked(
            events,
            event_kind,
            logical_attempt_id=physical.logical_attempt_id,
            physical_execution_id=physical.physical_execution_id,
            payload=payload,
        )
        events.append(event)

    def _append_halt_locked(
        self,
        events: list[ExecutionEventV1],
        reason: str,
        physical: PhysicalExecutionIdentityV1,
    ) -> None:
        event = self.store._append_unlocked(
            events,
            "fatal_integrity_halt",
            logical_attempt_id=physical.logical_attempt_id,
            physical_execution_id=physical.physical_execution_id,
            payload={"reason": redact_text(reason, self.store.redaction_secrets)},
        )
        events.append(event)

    def _terminal_payload(
        self,
        *,
        terminal_kind: TerminalKind,
        result_reference: str | None,
        cell_outcome: CellOutcome,
    ) -> dict[str, Any]:
        if terminal_kind not in _TERMINAL_KINDS:
            raise ValueError("unknown terminal kind")
        if cell_outcome.terminal_kind != terminal_kind:
            raise ValueError("cell outcome terminal kind does not match")
        expected_status = (
            "passed"
            if terminal_kind == "success"
            else "cancelled"
            if terminal_kind == "cancelled"
            else "failed"
        )
        if cell_outcome.status != expected_status:
            raise ValueError("cell outcome status disagrees with terminal kind")
        if terminal_kind in _BEHAVIORAL_TERMINAL:
            if result_reference is None or not result_reference.strip():
                raise ValueError(
                    "behavioral terminal requires a normalized result reference"
                )
        elif result_reference is not None and not result_reference.strip():
            raise ValueError("result reference cannot be empty")
        return {
            "terminal_kind": terminal_kind,
            "result_reference": (
                redact_text(result_reference, self.store.redaction_secrets)
                if result_reference is not None
                else None
            ),
            "cell_outcome": redact_value(
                _cell_outcome_record(cell_outcome),
                secrets=self.store.redaction_secrets,
            ),
        }

    @staticmethod
    def _validate_terminal_transition(
        state: PhysicalExecutionStateV1,
        terminal_kind: TerminalKind,
    ) -> None:
        if state.terminal_kind is not None:
            raise ExecutionRecoveryError("physical execution is already terminal")
        if (
            terminal_kind
            in {
                "success",
                "task_failure",
                "agent_timeout",
            }
            and not state.started
        ):
            raise ExecutionRecoveryError(
                "behavioral terminal evidence requires a started execution"
            )
        if (
            terminal_kind in {"sandbox_lost", "transport_interrupted"}
            and not state.started
        ):
            raise ExecutionRecoveryError(
                "post-start infrastructure evidence requires a started execution"
            )

    @staticmethod
    def _cleanup_payload(observation: CleanupObservationV1) -> dict[str, Any]:
        return {
            "verified": observation.verified,
            "scope_verified": observation.scope_verified,
            "post_run_inventory": observation.post_run_inventory,
            "discovered_resources": list(observation.discovered_resources),
            "inspected_resources": list(observation.inspected_resources),
            "removed_resources": list(observation.removed_resources),
            "remaining_resources": list(observation.remaining_resources),
            "receipt_reference": observation.receipt_reference,
        }

    @staticmethod
    def _validate_cleanup_resources(
        state: PhysicalExecutionStateV1,
        observation: CleanupObservationV1,
    ) -> None:
        expected_resources = (
            set(state.planned_harbor_resources)
            | set(state.observed_harbor_resources)
            | set(observation.discovered_resources)
        )
        if not expected_resources.issubset(observation.inspected_resources):
            raise ExecutionRecoveryError(
                "cleanup did not inspect every exact Harbor resource"
            )

    @classmethod
    def _validate_cleanup_transition(
        cls,
        state: PhysicalExecutionStateV1,
        observation: CleanupObservationV1,
    ) -> None:
        if state.terminal_kind is None or state.cleanup_observed:
            raise ExecutionRecoveryError("physical cleanup observation is out of order")
        cls._validate_cleanup_resources(state, observation)

    @staticmethod
    def _validate_cost_transition(
        state: PhysicalExecutionStateV1,
        *,
        actual_cost_micro_usd: int | None,
        authoritative: bool,
    ) -> None:
        if state.terminal_kind is None or state.cost_authoritative:
            raise ExecutionRecoveryError("physical cost observation is out of order")
        if state.cost_observed and not authoritative:
            raise ExecutionRecoveryError(
                "a missing cost may only be replaced by authoritative cost"
            )
        if authoritative and actual_cost_micro_usd is None:
            raise ValueError("authoritative cost cannot be missing")

    @staticmethod
    def _validate_canonical_transition(
        snapshot: ExecutionRecoverySnapshotV1,
        state: PhysicalExecutionStateV1,
        physical: PhysicalExecutionIdentityV1,
    ) -> None:
        if physical.logical_attempt_id in snapshot.canonical_results:
            raise ExecutionRecoveryError(
                "logical attempt already has a canonical result"
            )
        if not state.fully_reconciled:
            raise ExecutionRecoveryError("physical execution is not fully reconciled")
        if state.terminal_kind not in _BEHAVIORAL_TERMINAL:
            raise ExecutionRecoveryError(
                "only behavioral terminal evidence can be canonical"
            )
        if state.cell_outcome is None or not state.result_reference:
            raise ExecutionRecoveryError(
                "canonical result lacks a normalized terminal outcome"
            )


CellFinished = Callable[[PlannedCell, CellOutcome], None]
ResourceResolver = Callable[[PlannedCell], Sequence[str]]
CleanupVerifier = Callable[
    [PlannedCell, PhysicalExecutionIdentityV1, CellOutcome], CleanupObservationV1
]
CostResolver = Callable[
    [PlannedCell, PhysicalExecutionIdentityV1, CellOutcome], CostObservationV1
]
InterruptedExecutionReconciler = Callable[
    [PlannedCell, PhysicalExecutionIdentityV1],
    InterruptedExecutionObservationV1 | None,
]
CanonicalizationVerifier = Callable[
    [PlannedCell, PhysicalExecutionIdentityV1, CellOutcome],
    CanonicalizationObservationV1,
]
WaveBeginCell = Callable[[PlannedCell], Mapping[str, str] | None]
WaveFinishCell = Callable[[PlannedCell, CellOutcome], None]
WaveInvalidateCell = Callable[[PlannedCell, CellOutcome], None]
WaveFinalize = Callable[[tuple[CellOutcome, ...]], None]


@dataclass(frozen=True)
class AdmittedWaveLifecycle:
    """Host lifecycle attached only after physical work is durably admitted.

    A lifecycle may finalize local evidence after Agent work is terminal. Its
    callbacks must be idempotent because controller recovery can replay them.
    Canonical result selection remains in :class:`ExecutionRecoveryController`
    and happens only after ``finalize`` succeeds.
    """

    begin_cell: WaveBeginCell | None = None
    finish_cell: WaveFinishCell | None = None
    invalidate_cell: WaveInvalidateCell | None = None
    finalize: WaveFinalize | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.begin_cell, "wave begin callback"),
            (self.finish_cell, "wave finish callback"),
            (self.invalidate_cell, "wave invalidation callback"),
            (self.finalize, "wave finalize callback"),
        ):
            if value is not None and not callable(value):
                raise TypeError(f"{label} must be callable")


AdmittedWaveLifecycleFactory = Callable[
    [tuple[PlannedCell, ...]], AdmittedWaveLifecycle
]


@dataclass(frozen=True)
class ExecutionRecoveryAdapters:
    """Approved host-authoritative recovery boundary.

    The contract digest is part of the schedule identity. Resource discovery
    before a run is only an execution-start observation; cleanup authority
    always comes from the exact-scope post-run inventory in ``cleanup_verifier``.
    """

    contract_digest: str
    cleanup_verifier: CleanupVerifier
    cost_resolver: CostResolver
    canonicalization_verifier: CanonicalizationVerifier
    resource_resolver: ResourceResolver | None = None
    interrupted_reconciler: InterruptedExecutionReconciler | None = None
    redaction_secrets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_digest(self.contract_digest, "adapter contract digest")


def execute_recoverable_cells(  # noqa: C901 - bridges durable and cell lifecycles
    controller: ExecutionRecoveryController,
    cells: Sequence[PlannedCell],
    *,
    stage_authorization: StageExecutionAuthorizationV1,
    adapters: ExecutionRecoveryAdapters,
    repo_root: Path,
    runner: Callable[..., Any] | None = None,
    cell_finished: CellFinished | None = None,
    wave_lifecycle_factory: AdmittedWaveLifecycleFactory | None = None,
    cancellation_event: threading.Event | None = None,
) -> list[CellOutcome]:
    """Execute admitted waves through the canonical ``execute_cells`` path.

    The wrapper owns only execution governance. Candidate resolution, rendered
    jobs, cell execution, evidence callbacks, and normalized outcomes remain in
    their existing Fugue services.
    """

    if adapters.contract_digest != controller.schedule.adapter_contract_digest:
        raise ValueError("recovery adapter contract does not match the schedule")
    if stage_authorization.schedule_digest != controller.schedule.schedule_digest:
        raise ValueError("stage authorization does not match the schedule")
    by_attempt = {cell.attempt_id: cell for cell in cells}
    if set(by_attempt) != {
        item.logical_attempt_id for item in controller.schedule.logical_attempts
    }:
        raise ValueError("cells do not exactly cover the frozen execution schedule")
    if len(by_attempt) != len(cells):
        raise ValueError("cells contain duplicate logical attempts")
    run_ids = {cell.run_id for cell in cells}
    if len(run_ids) != 1:
        raise ValueError("recoverable cells must share one canonical run id")
    if any(not cell.applicable for cell in cells):
        raise ValueError("recoverable schedules may contain only applicable cells")
    stage_plans = tuple(
        item
        for item in controller.schedule.logical_attempts
        if item.stage_id == stage_authorization.stage_id
    )
    if len(stage_plans) != stage_authorization.maximum_logical_attempts:
        raise ValueError("authorized stage does not exactly match the frozen cells")
    stage_attempt_ids = {item.logical_attempt_id for item in stage_plans}
    cells_by_block: dict[str, list[PlannedCell]] = {}
    for plan in stage_plans:
        cells_by_block.setdefault(plan.admission_block_id, []).append(
            by_attempt[plan.logical_attempt_id]
        )
    for block_id, block_cells in cells_by_block.items():
        if len({cell.run_id for cell in block_cells}) != 1:
            raise ValueError(
                f"admission block {block_id} crosses canonical run boundaries"
            )
        destinations = {
            (
                cell.env.get("FUGUE_RESULT_EVIDENCE_PROJECT")
                or cell.env.get("FUGUE_WEAVE_PROJECT")
                or cell.env.get("WANDB_PROJECT")
                or ""
            )
            for cell in block_cells
        }
        if len(destinations) != 1:
            raise ValueError(
                f"admission block {block_id} crosses evidence destinations"
            )
    settled_physical_ids: set[str] = set()
    settlement_lock = threading.Lock()
    secrets = tuple(
        sorted(
            set(adapters.redaction_secrets)
            | {value for cell in cells for value in secrets_from_env(cell.env)},
            key=len,
            reverse=True,
        )
    )
    controller.activate_redaction_secrets(secrets)
    controller.authorize_stage(stage_authorization)

    def checked_cleanup(
        observation: CleanupObservationV1,
    ) -> CleanupObservationV1:
        _require_no_sensitive_text(
            controller._cleanup_payload(observation),
            secrets,
            "cleanup observation",
        )
        return observation

    def checked_cost(observation: CostObservationV1) -> CostObservationV1:
        _require_no_sensitive_text(
            {
                "source": observation.source,
                "receipt_reference": observation.receipt_reference,
            },
            secrets,
            "cost observation",
        )
        return observation

    def checked_canonicalization(
        observation: CanonicalizationObservationV1,
    ) -> CanonicalizationObservationV1:
        _require_no_sensitive_text(
            observation.to_dict(),
            secrets,
            "canonicalization observation",
        )
        return observation

    def identified_cell(
        physical: PhysicalExecutionIdentityV1,
    ) -> PlannedCell:
        cell = by_attempt[physical.logical_attempt_id]
        return replace(
            cell,
            physical_execution_id=physical.physical_execution_id,
            retry_ordinal=physical.retry_ordinal,
            env={
                **cell.env,
                "FUGUE_LOGICAL_ATTEMPT_ID": physical.logical_attempt_id,
                "FUGUE_PHYSICAL_EXECUTION_ID": physical.physical_execution_id,
                "FUGUE_RETRY_ORDINAL": str(physical.retry_ordinal),
            },
        )

    def physical_cell(
        physical: PhysicalExecutionIdentityV1,
    ) -> PlannedCell:
        return _rehydrate_physical_harbor_cell(
            identified_cell(physical),
            physical=physical,
            repo_root=repo_root,
        )

    def repair_partial_settlements(*, canonicalize: bool) -> None:
        snapshot = controller.snapshot()
        for state in snapshot.physical_executions.values():
            if state.terminal_kind is None or state.canonical:
                continue
            physical = state.identity
            cell = physical_cell(physical)
            outcome = _cell_outcome_from_record(state.cell_outcome)
            record_recovered_cell_outcome(repo_root, cell, outcome)
            if not state.cleanup_observed:
                try:
                    cleanup = checked_cleanup(
                        adapters.cleanup_verifier(cell, physical, outcome)
                    )
                    controller.cleanup(physical, cleanup)
                except Exception as exc:
                    controller.halt(
                        _safe_failure("cleanup reconciliation failed", exc, secrets),
                        logical_attempt_id=physical.logical_attempt_id,
                        physical_execution_id=physical.physical_execution_id,
                    )
                    continue
            state = controller.snapshot().physical_executions[
                physical.physical_execution_id
            ]
            if not state.cost_authoritative:
                try:
                    cost = checked_cost(adapters.cost_resolver(cell, physical, outcome))
                    if not cost.authoritative:
                        if not state.cost_observed:
                            controller.cost(physical, cost)
                        # Missing authoritative cost pauses admission. It is
                        # not an integrity failure and may be replaced later
                        # without rerunning the Agent.
                        continue
                    controller.cost(physical, cost)
                except Exception as exc:
                    controller.halt(
                        _safe_failure("cost reconciliation failed", exc, secrets),
                        logical_attempt_id=physical.logical_attempt_id,
                        physical_execution_id=physical.physical_execution_id,
                    )
                    continue
            state = controller.snapshot().physical_executions[
                physical.physical_execution_id
            ]
            if state.cost_lease_exceeded:
                if not controller.snapshot().fatal_reasons:
                    controller.halt(
                        f"{physical.logical_attempt_id}: cost exceeded its lease",
                        logical_attempt_id=physical.logical_attempt_id,
                        physical_execution_id=physical.physical_execution_id,
                    )
                continue
            if not state.cleanup_verified:
                if not controller.snapshot().fatal_reasons:
                    controller.halt(
                        f"{physical.logical_attempt_id}: cleanup_failure",
                        logical_attempt_id=physical.logical_attempt_id,
                        physical_execution_id=physical.physical_execution_id,
                    )
                continue
            if state.terminal_kind in controller.schedule.fatal_terminal_kinds:
                if not controller.snapshot().fatal_reasons:
                    controller.halt(
                        f"{physical.logical_attempt_id}: {state.terminal_kind}",
                        logical_attempt_id=physical.logical_attempt_id,
                        physical_execution_id=physical.physical_execution_id,
                    )
                continue
            if (
                canonicalize
                and state.terminal_kind in _BEHAVIORAL_TERMINAL
                and state.fully_reconciled
                and physical.logical_attempt_id
                not in controller.snapshot().canonical_results
            ):
                try:
                    reconciliation = checked_canonicalization(
                        adapters.canonicalization_verifier(cell, physical, outcome)
                    )
                    controller.select_canonical(physical, reconciliation)
                except ExecutionFinalizationPending:
                    raise
                except Exception as exc:
                    controller.halt(
                        _safe_failure(
                            "canonical evidence reconciliation failed", exc, secrets
                        ),
                        logical_attempt_id=physical.logical_attempt_id,
                        physical_execution_id=physical.physical_execution_id,
                    )

    def finalize_recovered_waves() -> None:
        """Retry host-only evidence finalization without Agent execution."""

        if wave_lifecycle_factory is None:
            return
        snapshot = controller.snapshot()
        pending_by_block: dict[str, list[PhysicalExecutionStateV1]] = {}
        for plan in stage_plans:
            if plan.logical_attempt_id in snapshot.canonical_results:
                continue
            states = [
                state
                for state in snapshot.physical_executions.values()
                if state.identity.logical_attempt_id == plan.logical_attempt_id
                and state.terminal_kind in _BEHAVIORAL_TERMINAL
                and state.fully_reconciled
            ]
            if not states:
                continue
            selected = max(states, key=lambda state: state.identity.retry_ordinal)
            pending_by_block.setdefault(plan.admission_block_id, []).append(selected)
        ordered_blocks = dict.fromkeys(
            plan.admission_block_id for plan in stage_plans
        )
        for block_id in ordered_blocks:
            states = pending_by_block.get(block_id)
            if not states:
                continue
            wave_cells = tuple(physical_cell(state.identity) for state in states)
            outcomes = tuple(
                _cell_outcome_from_record(state.cell_outcome) for state in states
            )
            lifecycle = wave_lifecycle_factory(wave_cells)
            if not isinstance(lifecycle, AdmittedWaveLifecycle):
                raise TypeError(
                    "wave lifecycle factory must return AdmittedWaveLifecycle"
                )
            try:
                if lifecycle.finish_cell is not None:
                    for cell, outcome in zip(wave_cells, outcomes, strict=True):
                        lifecycle.finish_cell(cell, outcome)
                if lifecycle.finalize is not None:
                    lifecycle.finalize(outcomes)
            except ExecutionFinalizationPending:
                raise
            except Exception as exc:
                first = states[0].identity
                controller.halt(
                    _safe_failure(
                        "recovered wave evidence finalization failed", exc, secrets
                    ),
                    logical_attempt_id=first.logical_attempt_id,
                    physical_execution_id=first.physical_execution_id,
                )
                return
            repair_partial_settlements(canonicalize=True)

    def reconcile_active_executions() -> None:
        snapshot = controller.snapshot()
        for physical_id in snapshot.active_physical_execution_ids:
            state = snapshot.physical_executions[physical_id]
            physical = state.identity
            if adapters.interrupted_reconciler is None:
                raise ExecutionRecoveryError(
                    "rehydrated controller has an unresolved active physical execution"
                )
            cell = physical_cell(physical)
            try:
                observation = adapters.interrupted_reconciler(cell, physical)
            except Exception as exc:
                controller.halt(
                    _safe_failure(
                        "active execution reconciliation failed", exc, secrets
                    ),
                    logical_attempt_id=physical.logical_attempt_id,
                    physical_execution_id=physical.physical_execution_id,
                )
                continue
            if observation is None:
                raise ExecutionRecoveryError(
                    "active physical execution lacks authoritative reconciliation"
                )
            checked_cleanup(observation.cleanup)
            checked_cost(observation.cost)
            reconciliation_reference = _verified_local_reference(
                repo_root=repo_root,
                reference=observation.reconciliation_receipt_reference,
                expected_sha256=observation.reconciliation_receipt_sha256,
                label="interrupted reconciliation receipt",
                secrets=secrets,
            )
            outcome = _redacted_outcome(observation.cell_outcome, secrets)
            if observation.terminal_kind in _BEHAVIORAL_TERMINAL:
                assert observation.result_reference is not None
                assert observation.result_sha256 is not None
                source_reference = _verified_local_reference(
                    repo_root=repo_root,
                    reference=observation.result_reference,
                    expected_sha256=observation.result_sha256,
                    label="interrupted behavioral result",
                    secrets=secrets,
                )
                expected_result = _absolute_path(cell.result_path, repo_root).resolve()
                observed_result = _absolute_path(
                    Path(source_reference), repo_root
                ).resolve()
                if observed_result != expected_result:
                    raise ExecutionRecoveryError(
                        "interrupted behavioral result is not the frozen cell result"
                    )
            else:
                source_reference = _archive_noncanonical_result(
                    repo_root=repo_root,
                    cell=cell,
                    physical=physical,
                )
            terminal_reference = _write_terminal_receipt(
                repo_root=repo_root,
                cell=cell,
                physical=physical,
                outcome=outcome,
                source_result_reference=source_reference,
                reconciliation_receipt_reference=(reconciliation_reference),
                reconciliation_receipt_sha256=(
                    observation.reconciliation_receipt_sha256
                ),
                secrets=secrets,
            )
            controller.reconcile_interrupted_physical(
                physical,
                terminal_kind=observation.terminal_kind,
                cell_outcome=outcome,
                result_reference=terminal_reference,
                cleanup=observation.cleanup,
                cost=observation.cost,
            )
            record_recovered_cell_outcome(repo_root, cell, outcome)

    while True:
        repair_partial_settlements(
            canonicalize=wave_lifecycle_factory is None,
        )
        reconcile_active_executions()
        repair_partial_settlements(
            canonicalize=wave_lifecycle_factory is None,
        )
        finalize_recovered_waves()
        if controller.snapshot().fatal_reasons:
            break
        admission = controller.admit(stage_id=stage_authorization.stage_id)
        if admission.status == "complete":
            break
        if admission.status == "blocked":
            snapshot = controller.snapshot()
            if snapshot.fatal_reasons:
                break
            if admission.reason in {
                "authoritative cost is missing",
                "physical executions are still active",
            }:
                raise ExecutionRecoveryPaused(admission.reason)
            raise ExecutionRecoveryError(admission.reason)
        physical_by_cell: dict[str, PhysicalExecutionIdentityV1] = {}
        wave_cells: list[PlannedCell] = []
        for physical in admission.physical_executions:
            cell = by_attempt[physical.logical_attempt_id]
            result_path = _absolute_path(cell.result_path, repo_root)
            if physical.retry_ordinal and result_path.exists():
                outcome = CellOutcome(
                    cell.id,
                    "failed",
                    error=(
                        "a prior physical execution left an unarchived result; "
                        "replacement execution was not started"
                    ),
                    terminal_kind="identity_drift",
                )
                controller.terminal(
                    physical,
                    terminal_kind="identity_drift",
                    result_reference=None,
                    cell_outcome=outcome,
                )
                continue
            physical_by_cell[cell.id] = physical
            wave_cells.append(
                _materialize_physical_harbor_cell(
                    identified_cell(physical),
                    physical=physical,
                    repo_root=repo_root,
                )
            )
        if not wave_cells:
            break
        wave_lifecycle = (
            wave_lifecycle_factory(tuple(wave_cells))
            if wave_lifecycle_factory is not None
            else AdmittedWaveLifecycle()
        )
        if not isinstance(wave_lifecycle, AdmittedWaveLifecycle):
            raise TypeError("wave lifecycle factory must return AdmittedWaveLifecycle")
        pending_finalizations: list[str] = []

        def begin_cell(
            cell: PlannedCell,
            *,
            _physical_by_cell: Mapping[str, PhysicalExecutionIdentityV1] = (
                physical_by_cell
            ),
            _wave_lifecycle: AdmittedWaveLifecycle = wave_lifecycle,
        ) -> Mapping[str, str] | None:
            physical = _physical_by_cell[cell.id]
            observed = tuple(
                adapters.resource_resolver(cell)
                if adapters.resource_resolver is not None
                else controller._plan_by_id[
                    physical.logical_attempt_id
                ].planned_harbor_resources
            )
            _require_no_sensitive_text(
                observed,
                secrets,
                "observed Harbor resources",
            )
            controller.started(physical, observed_harbor_resources=observed)
            return (
                _wave_lifecycle.begin_cell(cell)
                if _wave_lifecycle.begin_cell is not None
                else None
            )

        def settle_cell(
            cell: PlannedCell,
            outcome: CellOutcome,
            *,
            _physical_by_cell: Mapping[str, PhysicalExecutionIdentityV1] = (
                physical_by_cell
            ),
            _wave_lifecycle: AdmittedWaveLifecycle = wave_lifecycle,
            _pending_finalizations: list[str] = pending_finalizations,
        ) -> None:
            physical = _physical_by_cell[cell.id]
            with settlement_lock:
                if physical.physical_execution_id in settled_physical_ids:
                    return
                settled_physical_ids.add(physical.physical_execution_id)
            callback_failure = None
            finalization_pending = False
            lifecycle_callback = (
                _wave_lifecycle.invalidate_cell
                if outcome.terminal_kind in _RETRYABLE_INFRASTRUCTURE
                else _wave_lifecycle.finish_cell
            )
            if lifecycle_callback is not None:
                try:
                    lifecycle_callback(cell, outcome)
                except ExecutionFinalizationPending as exc:
                    finalization_pending = True
                    _pending_finalizations.append(
                        _safe_failure("host finalization is pending", exc, secrets)
                    )
                except Exception as exc:
                    callback_failure = _safe_failure(
                        "required wave evidence completion failed", exc, secrets
                    )
            if (
                cell_finished is not None
                and outcome.terminal_kind not in _RETRYABLE_INFRASTRUCTURE
                and callback_failure is None
                and not finalization_pending
            ):
                try:
                    cell_finished(cell, outcome)
                except Exception as exc:
                    callback_failure = _safe_failure(
                        "required evaluator completion failed", exc, secrets
                    )
            terminal_kind, durable_outcome = _normalized_terminal_outcome(
                outcome, callback_failure
            )
            durable_outcome = _redacted_outcome(durable_outcome, secrets)
            source_reference = (
                _existing_result_reference(repo_root, cell)
                if terminal_kind in _BEHAVIORAL_TERMINAL
                else _archive_noncanonical_result(
                    repo_root=repo_root,
                    cell=cell,
                    physical=physical,
                )
            )
            result_reference = _write_terminal_receipt(
                repo_root=repo_root,
                cell=cell,
                physical=physical,
                outcome=durable_outcome,
                source_result_reference=source_reference,
                reconciliation_receipt_reference=None,
                reconciliation_receipt_sha256=None,
                secrets=secrets,
            )
            try:
                cleanup = checked_cleanup(
                    adapters.cleanup_verifier(cell, physical, durable_outcome)
                )
            except Exception as exc:
                controller.terminal(
                    physical,
                    terminal_kind=terminal_kind,
                    result_reference=result_reference,
                    cell_outcome=durable_outcome,
                )
                controller.halt(
                    _safe_failure(
                        f"{physical.logical_attempt_id}: cleanup verifier failed",
                        exc,
                        secrets,
                    ),
                    logical_attempt_id=physical.logical_attempt_id,
                    physical_execution_id=physical.physical_execution_id,
                )
                return
            try:
                cost = checked_cost(
                    adapters.cost_resolver(cell, physical, durable_outcome)
                )
            except Exception as exc:
                controller.terminal(
                    physical,
                    terminal_kind=terminal_kind,
                    result_reference=result_reference,
                    cell_outcome=durable_outcome,
                )
                controller.cleanup(physical, cleanup)
                controller.halt(
                    _safe_failure(
                        f"{physical.logical_attempt_id}: cost resolver failed",
                        exc,
                        secrets,
                    ),
                    logical_attempt_id=physical.logical_attempt_id,
                    physical_execution_id=physical.physical_execution_id,
                )
                return
            controller.finalize(
                physical,
                terminal_kind=terminal_kind,
                result_reference=result_reference,
                cleanup=cleanup,
                cost=cost,
                cell_outcome=durable_outcome,
            )
            state = controller.snapshot().physical_executions[
                physical.physical_execution_id
            ]
            if (
                wave_lifecycle_factory is None
                and not finalization_pending
                and state.terminal_kind in _BEHAVIORAL_TERMINAL
                and state.fully_reconciled
            ):
                try:
                    reconciliation = checked_canonicalization(
                        adapters.canonicalization_verifier(
                            cell, physical, durable_outcome
                        )
                    )
                    controller.select_canonical(physical, reconciliation)
                except ExecutionFinalizationPending as exc:
                    _pending_finalizations.append(
                        _safe_failure("host finalization is pending", exc, secrets)
                    )
                except Exception as exc:
                    controller.halt(
                        _safe_failure(
                            "canonical evidence reconciliation failed", exc, secrets
                        ),
                        logical_attempt_id=physical.logical_attempt_id,
                        physical_execution_id=physical.physical_execution_id,
                    )

        def finish_cell(
            cell: PlannedCell,
            outcome: CellOutcome,
            *,
            _physical_by_cell: Mapping[str, PhysicalExecutionIdentityV1] = (
                physical_by_cell
            ),
        ) -> None:
            try:
                settle_cell(cell, outcome)
            except Exception as exc:
                physical = _physical_by_cell[cell.id]
                controller.halt(
                    f"{physical.logical_attempt_id}: settlement failed: "
                    f"{type(exc).__name__}: {exc}",
                    logical_attempt_id=physical.logical_attempt_id,
                    physical_execution_id=physical.physical_execution_id,
                )

        wave_outcomes = execute_cells(
            wave_cells,
            repo_root=repo_root,
            max_workers=len(wave_cells),
            runner=runner,
            cell_started=begin_cell,
            require_cell_started_success=True,
            cell_finished=finish_cell,
            cancellation_event=cancellation_event,
            redaction_secrets=secrets,
        )
        outcomes_by_cell = {item.cell_id: item for item in wave_outcomes}
        for cell in wave_cells:
            physical = physical_by_cell[cell.id]
            if (
                physical.physical_execution_id
                not in controller.snapshot().physical_executions
            ):
                raise ExecutionRecoveryError("physical lease disappeared")
            state = controller.snapshot().physical_executions[
                physical.physical_execution_id
            ]
            if state.terminal_kind is None:
                settle_cell(cell, outcomes_by_cell[cell.id])
        if pending_finalizations:
            raise ExecutionFinalizationPending(pending_finalizations[-1])
        behavioral_wave_outcomes = tuple(
            outcome
            for outcome in wave_outcomes
            if outcome.terminal_kind in _BEHAVIORAL_TERMINAL
        )
        if (
            wave_lifecycle_factory is not None
            and not controller.snapshot().fatal_reasons
        ):
            try:
                if wave_lifecycle.finalize is not None and behavioral_wave_outcomes:
                    wave_lifecycle.finalize(behavioral_wave_outcomes)
            except ExecutionFinalizationPending:
                raise
            except Exception as exc:
                first = admission.physical_executions[0]
                controller.halt(
                    _safe_failure("wave evidence finalization failed", exc, secrets),
                    logical_attempt_id=first.logical_attempt_id,
                    physical_execution_id=first.physical_execution_id,
                )
            else:
                repair_partial_settlements(canonicalize=True)
        if controller.snapshot().fatal_reasons:
            break
    snapshot = controller.snapshot()
    if snapshot.fatal_reasons:
        raise ExecutionRecoveryError(
            f"execution stopped at a fatal integrity gate: {snapshot.fatal_reasons[-1]}"
        )
    stage_canonical = stage_attempt_ids & set(snapshot.canonical_results)
    if len(stage_canonical) != len(stage_attempt_ids):
        raise ExecutionRecoveryError(
            "recoverable stage ended without one canonical result per cell"
        )
    canonical_outcomes: list[CellOutcome] = []
    for plan in controller.schedule.logical_attempts:
        if plan.stage_id != stage_authorization.stage_id:
            continue
        physical_id = snapshot.canonical_results.get(plan.logical_attempt_id)
        if physical_id is None:
            continue
        state = snapshot.physical_executions[physical_id]
        canonical_outcomes.append(_cell_outcome_from_record(state.cell_outcome))
    return canonical_outcomes


def _normalized_terminal_outcome(
    outcome: CellOutcome, callback_failure: str | None
) -> tuple[TerminalKind, CellOutcome]:
    if callback_failure is not None:
        durable = replace(
            outcome,
            status="failed",
            error=f"required evidence finalization failed: {callback_failure}",
            terminal_kind="evidence_failure",
        )
        return "evidence_failure", durable
    terminal_kind = outcome.terminal_kind
    if terminal_kind not in _TERMINAL_KINDS:
        durable = replace(
            outcome,
            status="failed",
            error="cell executor returned no typed terminal evidence",
            terminal_kind="execution_failure",
        )
        return "execution_failure", durable
    return terminal_kind, outcome  # type: ignore[return-value]


def _safe_failure(prefix: str, exc: BaseException, secrets: Sequence[str]) -> str:
    return redact_text(f"{prefix}: {type(exc).__name__}: {exc}", secrets)


def _contains_sensitive_text(value: Any, secrets: Sequence[str]) -> bool:
    if isinstance(value, str):
        return redact_text(value, secrets) != value
    if isinstance(value, Mapping):
        return any(_contains_sensitive_text(item, secrets) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive_text(item, secrets) for item in value)
    return False


def _require_no_sensitive_text(value: Any, secrets: Sequence[str], label: str) -> None:
    if _contains_sensitive_text(value, secrets):
        raise ExecutionRecoveryError(f"{label} contains sensitive data")


def _redacted_outcome(outcome: CellOutcome, secrets: Sequence[str]) -> CellOutcome:
    return replace(
        outcome,
        error=(redact_text(outcome.error, secrets) if outcome.error else None),
    )


def _existing_result_reference(repo_root: Path, cell: PlannedCell) -> str | None:
    path = _absolute_path(cell.result_path, repo_root)
    return _relative_reference(path, repo_root) if path.is_file() else None


def _verified_local_reference(
    *,
    repo_root: Path,
    reference: str,
    expected_sha256: str,
    label: str,
    secrets: Sequence[str],
) -> str:
    if redact_text(reference, secrets) != reference:
        raise ExecutionRecoveryError(f"{label} contains sensitive data")
    path = _absolute_path(Path(reference), repo_root).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ExecutionRecoveryError(f"{label} escapes the repository") from exc
    if not path.is_file():
        raise ExecutionRecoveryError(f"{label} does not exist")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ExecutionRecoveryError(f"{label} digest does not match")
    return _relative_reference(path, repo_root)


def _write_terminal_receipt(
    *,
    repo_root: Path,
    cell: PlannedCell,
    physical: PhysicalExecutionIdentityV1,
    outcome: CellOutcome,
    source_result_reference: str | None,
    reconciliation_receipt_reference: str | None,
    reconciliation_receipt_sha256: str | None,
    secrets: Sequence[str],
) -> str:
    source_sha256 = None
    if source_result_reference:
        source = Path(source_result_reference)
        if not source.is_absolute():
            source = repo_root / source
        if source.is_file():
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    unsigned = redact_value(
        {
            "schema_version": 1,
            "logical_attempt_id": physical.logical_attempt_id,
            "physical_execution_id": physical.physical_execution_id,
            "retry_ordinal": physical.retry_ordinal,
            "cell_id": cell.id,
            "cell_outcome": _cell_outcome_record(outcome),
            "source_result_reference": source_result_reference,
            "source_result_sha256": source_sha256,
            "reconciliation_receipt_reference": (reconciliation_receipt_reference),
            "reconciliation_receipt_sha256": reconciliation_receipt_sha256,
        },
        secrets=secrets,
    )
    payload = {**unsigned, "receipt_digest": stable_digest(unsigned)}
    destination = (
        repo_root
        / ".fugue"
        / "runtime"
        / cell.run_id
        / "physical-executions"
        / physical.physical_execution_id
        / "terminal.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ExecutionRecoveryError(
                "existing physical terminal receipt is unreadable"
            ) from exc
        if existing != payload:
            raise ExecutionRecoveryError(
                "physical terminal receipt changed during recovery"
            )
    else:
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()
    return _relative_reference(destination, repo_root)


def _relative_reference(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _cell_outcome_record(outcome: CellOutcome) -> dict[str, Any]:
    return {
        "cell_id": outcome.cell_id,
        "status": outcome.status,
        "returncode": outcome.returncode,
        "error": outcome.error,
        "benchmark_outcome": outcome.benchmark_outcome,
        "reward": outcome.reward,
        "runtime_outcome": outcome.runtime_outcome,
        "terminal_kind": outcome.terminal_kind,
    }


def _cell_outcome_from_record(
    raw: Mapping[str, Any] | None,
) -> CellOutcome:
    if raw is None:
        raise ValueError("cell outcome record is required")
    _require_fields(
        raw,
        {
            "cell_id",
            "status",
            "returncode",
            "error",
            "benchmark_outcome",
            "reward",
            "runtime_outcome",
            "terminal_kind",
        },
        "cell outcome",
    )
    status = _strict_str(raw.get("status"), "cell_outcome.status")
    benchmark_outcome = _strict_str(
        raw.get("benchmark_outcome"), "cell_outcome.benchmark_outcome"
    )
    runtime_outcome = _strict_str(
        raw.get("runtime_outcome"), "cell_outcome.runtime_outcome"
    )
    terminal_kind = _strict_str(raw.get("terminal_kind"), "cell_outcome.terminal_kind")
    if terminal_kind not in _TERMINAL_KINDS:
        raise ValueError("cell terminal kind is invalid")
    if status not in {
        "pending",
        "running",
        "passed",
        "failed",
        "not_applicable",
        "cancelled",
        "interrupted",
    }:
        raise ValueError("cell outcome status is invalid")
    if benchmark_outcome not in {
        "passed",
        "failed",
        "unscored",
        "not_applicable",
    }:
        raise ValueError("cell benchmark outcome is invalid")
    if runtime_outcome not in {
        "completed",
        "timed_out",
        "cancelled",
        "not_started",
        "not_applicable",
    }:
        raise ValueError("cell runtime outcome is invalid")
    reward = float(raw["reward"]) if raw.get("reward") is not None else None
    if reward is not None and not math.isfinite(reward):
        raise ValueError("cell reward must be finite")
    return CellOutcome(
        cell_id=_strict_str(raw.get("cell_id"), "cell_outcome.cell_id"),
        status=status,  # type: ignore[arg-type]
        returncode=(
            _strict_int(raw.get("returncode"), "cell_outcome.returncode")
            if raw.get("returncode") is not None
            else None
        ),
        error=(
            _strict_str(raw.get("error"), "cell_outcome.error")
            if raw.get("error") is not None
            else None
        ),
        benchmark_outcome=benchmark_outcome,  # type: ignore[arg-type]
        reward=reward,
        runtime_outcome=runtime_outcome,  # type: ignore[arg-type]
        terminal_kind=terminal_kind,  # type: ignore[arg-type]
    )


def _absolute_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _materialize_physical_harbor_cell(  # noqa: C901 - strict projection audit
    cell: PlannedCell,
    *,
    physical: PhysicalExecutionIdentityV1,
    repo_root: Path,
    _write_once: bool = True,
) -> PlannedCell:
    """Give one physical Harbor execution a fresh immutable state namespace."""

    if not cell.command or Path(str(cell.command[0])).name != "harbor":
        return cell
    command = list(cell.command)
    config_positions = [
        index for index, value in enumerate(command) if value == "--config"
    ]
    if len(config_positions) != 1 or config_positions[0] + 1 >= len(command):
        raise ExecutionRecoveryError(
            "recoverable Harbor command must contain exactly one --config path"
        )
    logical_input = _absolute_path(cell.config_path, repo_root)
    if logical_input.is_symlink():
        raise ExecutionRecoveryError("approved logical Harbor config is a symlink")
    logical_config = logical_input.resolve()
    try:
        logical_reference = logical_config.relative_to(repo_root.resolve()).as_posix()
        logical_bytes = logical_config.read_bytes()
        raw = json.loads(logical_bytes)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise ExecutionRecoveryError(
            "approved logical Harbor config is unavailable or invalid"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ExecutionRecoveryError("approved logical Harbor config is not an object")
    logical_digest = hashlib.sha256(logical_bytes).hexdigest()
    if cell.config_sha256 and cell.config_sha256 != logical_digest:
        raise ExecutionRecoveryError("approved logical Harbor config changed")
    original_job_name = raw.get("job_name")
    if not isinstance(original_job_name, str) or not original_job_name.strip():
        raise ExecutionRecoveryError("approved Harbor config has no job name")
    fugue_config = raw.get("fugue")
    if not isinstance(fugue_config, Mapping):
        raise ExecutionRecoveryError("approved Harbor config has no Fugue identity")
    agents = raw.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ExecutionRecoveryError("approved Harbor config has no Agent")

    suffix = f"-p{physical.physical_execution_id[:12]}"
    maximum_base = 120 - len(suffix)
    physical_job_name = f"{original_job_name[:maximum_base].rstrip('-')}{suffix}"
    physical_root = (
        repo_root.resolve()
        / ".fugue"
        / "runtime"
        / cell.run_id
        / "physical-executions"
        / physical.physical_execution_id
        / "harbor"
    )
    physical_jobs_dir = physical_root / "jobs"
    physical_config_path = physical_root / "config.json"
    physical_result_path = physical_jobs_dir / physical_job_name / "result.json"
    physical_parent = physical_root.parent
    if _write_once:
        physical_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if physical_parent.is_symlink() or physical_root.exists():
            raise ExecutionRecoveryError(
                "physical Harbor namespace is not a fresh write-once directory"
            )
        physical_root.mkdir(mode=0o700)
        os.chmod(physical_root, 0o700)
    elif (
        not physical_root.is_dir()
        or physical_root.is_symlink()
        or physical_parent.is_symlink()
    ):
        raise ExecutionRecoveryError(
            "journaled physical Harbor namespace is unavailable"
        )

    namespace = {
        "schema_version": 1,
        "logical_attempt_id": physical.logical_attempt_id,
        "physical_execution_id": physical.physical_execution_id,
        "retry_ordinal": physical.retry_ordinal,
        "logical_config_path": logical_reference,
        "logical_config_sha256": logical_digest,
        "harbor_job_name": physical_job_name,
        "harbor_jobs_dir": physical_jobs_dir.as_posix(),
        "harbor_result_path": physical_result_path.as_posix(),
        # These are container-local paths. Their host backing directory is
        # unique because the Harbor job/output namespace above is unique.
        "claude_config_dir": "/logs/agent/sessions",
        "claude_plugin_cache_dir": "/logs/agent/sessions/plugin-cache",
    }
    namespace = {**namespace, "namespace_digest": stable_digest(namespace)}
    agent_environment = {
        "FUGUE_LOGICAL_ATTEMPT_ID": physical.logical_attempt_id,
        "FUGUE_PHYSICAL_EXECUTION_ID": physical.physical_execution_id,
        "FUGUE_RETRY_ORDINAL": str(physical.retry_ordinal),
        "FUGUE_PHYSICAL_AGENT_STATE_NAMESPACE": namespace["namespace_digest"],
    }
    physical_agents: list[dict[str, Any]] = []
    for raw_agent in agents:
        if not isinstance(raw_agent, Mapping):
            raise ExecutionRecoveryError("approved Harbor Agent config is invalid")
        raw_env = raw_agent.get("env") or {}
        if not isinstance(raw_env, Mapping):
            raise ExecutionRecoveryError("approved Harbor Agent environment is invalid")
        physical_agents.append(
            {
                **dict(raw_agent),
                "env": {**dict(raw_env), **agent_environment},
            }
        )
    physical_config = {
        **dict(raw),
        "job_name": physical_job_name,
        "jobs_dir": physical_jobs_dir.as_posix(),
        "agents": physical_agents,
        "fugue": {
            **dict(fugue_config),
            "physical_execution": namespace,
        },
    }
    if physical_config_path.is_symlink():
        raise ExecutionRecoveryError("physical Harbor config cannot be a symlink")
    if physical_jobs_dir.exists() and physical_jobs_dir.is_symlink():
        raise ExecutionRecoveryError("physical Harbor jobs directory is a symlink")
    if physical_config_path.is_file():
        try:
            existing = json.loads(physical_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutionRecoveryError(
                "physical Harbor config is unreadable"
            ) from exc
        if existing != physical_config:
            raise ExecutionRecoveryError("physical Harbor config changed")
    elif _write_once:
        atomic_write_json(physical_config_path, physical_config)
    else:
        raise ExecutionRecoveryError(
            "journaled physical Harbor config is unavailable"
        )
    physical_digest = hashlib.sha256(physical_config_path.read_bytes()).hexdigest()
    command[config_positions[0] + 1] = physical_config_path.as_posix()
    forbidden = {
        "--plugin",
        "--pk",
        "--plugin-kwarg",
        "--upload",
    }
    if any(value in forbidden for value in command):
        raise ExecutionRecoveryError(
            "recoverable Harbor commands cannot supply plugins or uploads"
        )
    terminal_receipt = physical_parent / "harbor-terminal.json"
    plugin_kwargs = {
        "logical_attempt_id": physical.logical_attempt_id,
        "physical_execution_id": physical.physical_execution_id,
        "retry_ordinal": str(physical.retry_ordinal),
        "cell_id": cell.id,
        "config_path": physical_config_path.as_posix(),
        "config_sha256": physical_digest,
        "result_path": physical_result_path.as_posix(),
        "receipt_path": terminal_receipt.as_posix(),
    }
    command.extend(
        ["--plugin", "fugue.bench.harbor_terminal:DurableHarborTerminalPlugin"]
    )
    for key, value in plugin_kwargs.items():
        command.extend(["--pk", f"{key}={value}"])
    return replace(
        cell,
        config_path=physical_config_path,
        result_path=physical_result_path,
        command=tuple(command),
        config_sha256=physical_digest,
        env={
            **cell.env,
            **agent_environment,
            "FUGUE_HARBOR_CONFIG": physical_config_path.as_posix(),
            "FUGUE_JOB_NAME": physical_job_name,
            "FUGUE_PHYSICAL_HARBOR_JOBS_DIR": physical_jobs_dir.as_posix(),
            "FUGUE_PHYSICAL_RESULT_PATH": physical_result_path.as_posix(),
        },
    )


def _rehydrate_physical_harbor_cell(
    cell: PlannedCell,
    *,
    physical: PhysicalExecutionIdentityV1,
    repo_root: Path,
) -> PlannedCell:
    """Read and recompute an existing physical namespace without writing."""

    return _materialize_physical_harbor_cell(
        cell,
        physical=physical,
        repo_root=repo_root,
        _write_once=False,
    )


def _archive_noncanonical_result(
    *,
    repo_root: Path,
    cell: PlannedCell,
    physical: PhysicalExecutionIdentityV1,
) -> str | None:
    source = _absolute_path(cell.result_path, repo_root)
    destination = (
        repo_root
        / ".fugue"
        / "runtime"
        / cell.run_id
        / "physical-executions"
        / physical.physical_execution_id
        / source.name
    )
    if not source.is_file():
        if not destination.is_file() or destination.is_symlink():
            return None
        if not _runner_receipt_binds_archived_result(
            repo_root=repo_root,
            cell=cell,
            physical=physical,
            source=source,
            destination=destination,
        ):
            raise ExecutionRecoveryError(
                "existing physical result archive lacks its runner binding"
            )
        return _relative_reference(destination, repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.exists():
            raise ExecutionRecoveryError(
                "physical execution result archive already exists"
            )
        return _relative_reference(destination, repo_root)
    os.replace(source, destination)
    return _relative_reference(destination, repo_root)


def _runner_receipt_binds_archived_result(
    *,
    repo_root: Path,
    cell: PlannedCell,
    physical: PhysicalExecutionIdentityV1,
    source: Path,
    destination: Path,
) -> bool:
    receipt_path = destination.parent / "runner-terminal.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(receipt, Mapping):
        return False
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_digest"
    }
    try:
        source_reference = source.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return False
    return bool(
        receipt.get("receipt_digest") == stable_digest(unsigned)
        and receipt.get("logical_attempt_id") == physical.logical_attempt_id
        and receipt.get("physical_execution_id")
        == physical.physical_execution_id
        and receipt.get("retry_ordinal") == physical.retry_ordinal
        and receipt.get("config_sha256") == cell.config_sha256
        and receipt.get("result_reference") == source_reference
        and receipt.get("result_sha256")
        == hashlib.sha256(destination.read_bytes()).hexdigest()
    )


def _require_fields(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        messages = []
        if unknown:
            messages.append("unknown=" + ",".join(unknown))
        if missing:
            messages.append("missing=" + ",".join(missing))
        raise ValueError(f"{label} fields are invalid ({'; '.join(messages)})")


def _require_safe_id(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} is invalid")


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a sha256 digest")


def _require_positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _require_nonnegative_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _strict_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _strict_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _strict_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value
