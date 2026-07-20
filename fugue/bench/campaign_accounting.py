from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CostAccounting:
    observed_cost_usd: float
    accounted_cost_usd: float
    measured_cells: int
    unmeasured_cells: int
    maximum_measured_cell_cost_usd: float | None
    fallback_cell_cost_usd: float


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
    return max(values) if values else None


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
        fallback_cell_cost_usd=fallback,
    )


def _non_negative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result
