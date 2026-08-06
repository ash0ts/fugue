from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json


class BudgetAdmissionPaused(RuntimeError):
    """Admission must wait for authoritative cost/accounting evidence."""

    admission_paused = True


class BudgetSettlementExceeded(RuntimeError):
    """Observed spend exceeded an immutable execution or study ceiling."""


@dataclass(frozen=True)
class CostAccounting:
    observed_cost_usd: float
    accounted_cost_usd: float
    measured_cells: int
    unmeasured_cells: int
    maximum_measured_cell_cost_usd: float | None


@dataclass(frozen=True)
class BudgetLeaseV1:
    physical_execution_id: str
    reserved_cost_usd: float
    status: str
    actual_cost_usd: float | None = None
    lease_digest: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"active", "settled", "missing_cost", "overspent"}:
            raise ValueError("unknown budget lease status")
        _non_negative(self.reserved_cost_usd, "reserved lease cost")
        if self.actual_cost_usd is not None:
            _non_negative(self.actual_cost_usd, "actual lease cost")
        if self.status in {"settled", "overspent"} and self.actual_cost_usd is None:
            raise ValueError("terminal budget lease requires actual cost")
        unsigned = {**asdict(self), "lease_digest": ""}
        digest = stable_digest(unsigned)
        if self.lease_digest and self.lease_digest != digest:
            raise ValueError("budget lease digest does not match")
        object.__setattr__(self, "lease_digest", digest)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BudgetLeaseLedger:
    """Small locked ledger that reserves spend before physical execution."""

    def __init__(
        self,
        path: Path,
        *,
        maximum_total_cost_usd: float,
        maximum_in_flight_cost_usd: float,
        maximum_in_flight_executions: int,
        maximum_physical_executions: int,
    ) -> None:
        self.path = path
        self.maximum_total_cost_usd = _non_negative(
            maximum_total_cost_usd, "maximum total cost"
        )
        self.maximum_in_flight_cost_usd = _non_negative(
            maximum_in_flight_cost_usd, "maximum in-flight cost"
        )
        self.maximum_in_flight_executions = int(maximum_in_flight_executions)
        self.maximum_physical_executions = int(maximum_physical_executions)
        if self.maximum_in_flight_executions < 1:
            raise ValueError("maximum in-flight executions must be positive")
        if self.maximum_physical_executions < 1:
            raise ValueError("maximum physical executions must be positive")
        if self.maximum_in_flight_cost_usd > self.maximum_total_cost_usd:
            raise ValueError("in-flight cost exceeds total cost ceiling")

    def acquire(
        self, physical_execution_id: str, *, reserved_cost_usd: float
    ) -> BudgetLeaseV1:
        reserve = _non_negative(reserved_cost_usd, "reserved lease cost")
        with self._lock():
            state = self._read()
            leases = self._leases(state)
            prior = leases.get(physical_execution_id)
            if prior is not None:
                lease = self._lease(prior)
                if lease.reserved_cost_usd != reserve:
                    raise ValueError("budget lease identity was reused with new input")
                return lease
            if any(item.get("status") == "overspent" for item in leases.values()):
                raise BudgetSettlementExceeded(
                    "observed spend exceeded an immutable budget ceiling"
                )
            if any(item.get("status") == "missing_cost" for item in leases.values()):
                raise BudgetAdmissionPaused(
                    "missing cost pauses later physical executions"
                )
            if len(leases) >= self.maximum_physical_executions:
                raise RuntimeError("physical execution ceiling exceeded")
            active = [item for item in leases.values() if item.get("status") == "active"]
            if len(active) >= self.maximum_in_flight_executions:
                raise BudgetAdmissionPaused("in-flight execution ceiling exceeded")
            active_cost = sum(float(item["reserved_cost_usd"]) for item in active)
            if active_cost + reserve > self.maximum_in_flight_cost_usd + 1e-9:
                raise BudgetAdmissionPaused("in-flight cost ceiling exceeded")
            accounted = sum(
                float(item.get("actual_cost_usd") or 0)
                for item in leases.values()
                if item.get("status") in {"settled", "overspent"}
            )
            if accounted + active_cost + reserve > self.maximum_total_cost_usd + 1e-9:
                raise RuntimeError("total cost ceiling exceeded")
            lease = BudgetLeaseV1(physical_execution_id, reserve, "active")
            leases[physical_execution_id] = lease.to_dict()
            self._write(state)
            return lease

    def settle(
        self, physical_execution_id: str, *, actual_cost_usd: float | None
    ) -> BudgetLeaseV1:
        with self._lock():
            state = self._read()
            leases = self._leases(state)
            if physical_execution_id not in leases:
                raise ValueError("budget lease does not exist")
            prior = self._lease(leases[physical_execution_id])
            if prior.status in {"settled", "overspent"}:
                if prior.actual_cost_usd != actual_cost_usd:
                    raise ValueError("terminal budget lease cannot change cost")
                if prior.status == "overspent":
                    raise BudgetSettlementExceeded(
                        "observed spend exceeded an immutable budget ceiling"
                    )
                return prior
            observed = (
                _non_negative(actual_cost_usd, "actual lease cost")
                if actual_cost_usd is not None
                else None
            )
            other_accounted = sum(
                float(item.get("actual_cost_usd") or 0)
                for key, item in leases.items()
                if key != physical_execution_id
                and item.get("status") in {"settled", "overspent"}
            )
            violations: list[str] = []
            if observed is not None:
                if observed > prior.reserved_cost_usd + 1e-9:
                    violations.append("per-execution reserve")
                if other_accounted + observed > self.maximum_total_cost_usd + 1e-9:
                    violations.append("total cost ceiling")
            lease = BudgetLeaseV1(
                physical_execution_id,
                prior.reserved_cost_usd,
                "missing_cost"
                if observed is None
                else "overspent"
                if violations
                else "settled",
                observed,
            )
            leases[physical_execution_id] = lease.to_dict()
            self._write(state)
            if violations:
                raise BudgetSettlementExceeded(
                    "observed spend exceeded immutable " + " and ".join(violations)
                )
            return lease

    def release_unstarted(self, physical_execution_id: str) -> None:
        """Release an admission lease when no physical process was launched."""

        with self._lock():
            state = self._read()
            leases = self._leases(state)
            prior = leases.get(physical_execution_id)
            if prior is None:
                return
            if self._lease(prior).status != "active":
                raise ValueError("only an unstarted active lease may be released")
            del leases[physical_execution_id]
            self._write(state)

    def snapshot(self) -> dict[str, Any]:
        with self._lock():
            return self._read()

    def _lock(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(f"{self.path}.lock")

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": 1,
                "policy": self._policy(),
                "leases": {},
            }
        import json

        value = json.loads(self.path.read_text())
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("invalid budget lease ledger")
        if value.get("policy") != self._policy():
            raise ValueError("budget lease ledger policy changed")
        supplied = str(value.get("ledger_digest") or "")
        unsigned = {
            "schema_version": 1,
            "policy": value["policy"],
            "leases": value.get("leases"),
        }
        if not supplied or stable_digest(unsigned) != supplied:
            raise ValueError("budget lease ledger digest does not match")
        self._leases(value)
        return value

    def _write(self, state: dict[str, Any]) -> None:
        state["ledger_digest"] = stable_digest(
            {
                "schema_version": 1,
                "policy": state["policy"],
                "leases": state["leases"],
            }
        )
        atomic_write_json(self.path, state)

    def _policy(self) -> dict[str, Any]:
        return {
            "maximum_total_cost_usd": self.maximum_total_cost_usd,
            "maximum_in_flight_cost_usd": self.maximum_in_flight_cost_usd,
            "maximum_in_flight_executions": self.maximum_in_flight_executions,
            "maximum_physical_executions": self.maximum_physical_executions,
        }

    @staticmethod
    def _leases(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw = state.setdefault("leases", {})
        if not isinstance(raw, dict):
            raise ValueError("invalid budget lease ledger entries")
        return raw

    @staticmethod
    def _lease(raw: Mapping[str, Any]) -> BudgetLeaseV1:
        return BudgetLeaseV1(
            physical_execution_id=str(raw.get("physical_execution_id") or ""),
            reserved_cost_usd=float(raw.get("reserved_cost_usd") or 0),
            status=str(raw.get("status") or ""),
            actual_cost_usd=(
                float(raw["actual_cost_usd"])
                if raw.get("actual_cost_usd") is not None
                else None
            ),
            lease_digest=str(raw.get("lease_digest") or ""),
        )


def settle_execution_budget_leases(
    ledgers: Sequence[BudgetLeaseLedger],
    *,
    physical_execution_id: str,
    terminal_kind: str | None,
    runtime_outcome: str | None,
    actual_cost_usd: float | None,
) -> None:
    """Settle spend while retaining every admitted physical execution."""

    zero_cost_prestart_kinds = {
        "runner_start_failure",
        "evidence_initialization_failure",
        "prestart_cancelled",
        "pre_provider_failure",
    }
    if terminal_kind in zero_cost_prestart_kinds and runtime_outcome == "not_started":
        for ledger in ledgers:
            ledger.settle(physical_execution_id, actual_cost_usd=0.0)
        return
    failures: list[str] = []
    for ledger in ledgers:
        try:
            ledger.settle(
                physical_execution_id,
                actual_cost_usd=actual_cost_usd,
            )
        except BudgetSettlementExceeded as exc:
            # Preserve the observed overspend in every applicable ledger before
            # surfacing the fatal accounting gate to the execution lifecycle.
            failures.append(str(exc))
    if failures:
        raise BudgetSettlementExceeded("; ".join(dict.fromkeys(failures)))


def measured_row_cost(row: Mapping[str, Any]) -> float | None:
    """Return the conservative measured cost without inventing a public value."""

    interaction = row.get("task_interaction")
    if (
        isinstance(interaction, Mapping)
        and int(interaction.get("unmeasured_paid_calls") or 0)
        and interaction.get("accounted_interactor_cost_usd") is None
    ):
        return None

    values: list[float] = []
    for field in ("cost_usd", "weave_total_cost_usd"):
        raw = row.get(field)
        if raw is None:
            continue
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError("measured campaign costs must be finite and non-negative")
        values.append(value)
    if (
        values
        and isinstance(interaction, Mapping)
        and row.get("agent_cost_usd") is None
        and isinstance(interaction.get("accounted_interactor_cost_usd"), (int, float))
    ):
        interactor_cost = _non_negative(
            float(interaction["accounted_interactor_cost_usd"]),
            "accounted interactor cost",
        )
        values = [value + interactor_cost for value in values]
    if not values:
        # An accounted judge reserve cannot make an otherwise unmeasured Agent
        # execution measurable. Keep admission paused until the authoritative
        # Agent cost is available.
        return None
    judge_cost = row.get("comparison_judge_accounted_cost_usd")
    if judge_cost is not None:
        judge_cost = _non_negative(float(judge_cost), "accounted judge cost")
    return max(values) + float(judge_cost or 0.0)


def reserve_campaign_cost(
    *,
    cell_count: int,
    initial_cell_reserve_usd: float,
    safety_margin: float,
    prior_maximum_cell_cost_usd: float | None = None,
) -> tuple[float, float]:
    """Return total and per-cell admission reservations."""

    if cell_count < 1:
        raise ValueError("campaign admission requires at least one cell")
    initial = _non_negative(initial_cell_reserve_usd, "initial cell reserve")
    margin = float(safety_margin)
    if not math.isfinite(margin) or margin < 1:
        raise ValueError("campaign safety margin must be finite and at least one")
    observed = (
        _non_negative(prior_maximum_cell_cost_usd, "prior maximum cell cost")
        if prior_maximum_cell_cost_usd is not None
        else 0.0
    )
    per_cell = max(initial, observed * margin)
    return per_cell * cell_count, per_cell


def account_prediction_costs(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_cells: int,
    reserved_cell_cost_usd: float,
) -> CostAccounting:
    """Account missing cell costs conservatively while keeping rows unchanged."""

    if expected_cells < 1:
        raise ValueError("expected campaign cells must be positive")
    reserved = _non_negative(reserved_cell_cost_usd, "reserved cell cost")
    costs = [measured_row_cost(row) for row in rows]
    measured = [value for value in costs if value is not None]
    maximum = max(measured) if measured else None
    fallback = max(reserved, maximum or 0.0)
    missing = max(0, expected_cells - len(measured))
    return CostAccounting(
        observed_cost_usd=sum(measured),
        accounted_cost_usd=sum(measured) + missing * fallback,
        measured_cells=len(measured),
        unmeasured_cells=missing,
        maximum_measured_cell_cost_usd=maximum,
    )


def _non_negative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result
