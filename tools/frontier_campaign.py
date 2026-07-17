#!/usr/bin/env python3
"""Admit frontier cohorts against a durable, conservative campaign budget."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fugue.bench.files import atomic_write_json

SCHEMA_VERSION = 1


def _measured_row_cost(row: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for field in ("cost_usd", "weave_total_cost_usd"):
        raw = row.get(field)
        if raw is None:
            continue
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError("measured campaign costs must be finite and non-negative")
        values.append(value)
    return max(values) if values else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"row {number} must be an object")
        rows.append(value)
    return rows


def validate_canary(
    rows: Sequence[Mapping[str, Any]], *, expected_predictions: int, model: str
) -> dict[str, float | int]:
    if len(rows) != expected_predictions:
        raise ValueError(
            f"canary has {len(rows)} rows; expected {expected_predictions}"
        )
    prediction_ids: set[str] = set()
    measured_costs: list[float] = []
    for row in rows:
        if row.get("schema_version") != 1 or row.get("prediction_schema_version") != 1:
            raise ValueError("canary rows must use canonical schema 1")
        prediction_id = str(row.get("prediction_id") or "")
        if not prediction_id or prediction_id in prediction_ids:
            raise ValueError("canary prediction IDs must be non-empty and unique")
        prediction_ids.add(prediction_id)
        if row.get("execution_kind") != "agent":
            raise ValueError("frontier canaries may contain only Agent predictions")
        if row.get("model") != model:
            raise ValueError("canary row model does not match the admitted cohort")
        if str(row.get("status") or "") != "passed":
            raise ValueError("canary rows must complete without execution failure")
        outcome = row.get("adapter_outcome")
        if isinstance(outcome, Mapping):
            execution = outcome.get("execution")
            if not isinstance(execution, Mapping) or execution.get("state") != "completed":
                raise ValueError("canary rows must have completed adapter execution")
        if str(row.get("trace_link_status") or row.get("agent_link_status") or "") not in {
            "verified",
            "linked",
            "exact",
        }:
            raise ValueError("canary rows must have exact Agent links")
        root_ids = row.get("weave_root_span_ids")
        if (
            not isinstance(root_ids, list)
            or len(root_ids) != 1
            or not isinstance(root_ids[0], str)
            or not root_ids[0]
            or row.get("root_span_id") != root_ids[0]
            or int(row.get("weave_turn_count") or 0) != 1
        ):
            raise ValueError("each canary prediction must have exactly one Agent root")
        conversation_ids = row.get("weave_conversation_ids")
        if (
            not isinstance(conversation_ids, list)
            or len(conversation_ids) != 1
            or not isinstance(conversation_ids[0], str)
            or not conversation_ids[0]
            or row.get("observed_conversation_id") != conversation_ids[0]
        ):
            raise ValueError("each canary prediction must have exactly one conversation")
        cost = _measured_row_cost(row)
        if cost is not None:
            measured_costs.append(cost)
    if not measured_costs:
        raise ValueError("cannot forecast a cohort without any measured canary cost")
    maximum = max(measured_costs)
    return {
        "predictions": len(rows),
        "measured_predictions": len(measured_costs),
        "observed_cost_usd": sum(measured_costs),
        "accounted_cost_usd": sum(measured_costs)
        + (len(rows) - len(measured_costs)) * maximum,
        "maximum_measured_cell_cost_usd": maximum,
    }


def load_ledger(path: Path, *, cap_usd: float) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "cap_usd": cap_usd, "cohorts": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("campaign ledger must use schema 1")
    if float(value.get("cap_usd") or 0) != cap_usd:
        raise ValueError("campaign cap does not match the existing ledger")
    cohorts = value.get("cohorts")
    if not isinstance(cohorts, list):
        raise ValueError("campaign ledger cohorts must be a list")
    return value


def admit_cohort(
    ledger: Mapping[str, Any],
    *,
    cohort_id: str,
    model: str,
    canary: Mapping[str, float | int],
    cohort_predictions: int,
    safety_margin: float,
) -> dict[str, Any]:
    if not cohort_id or cohort_predictions < 1 or safety_margin < 1:
        raise ValueError("cohort admission inputs are invalid")
    existing = [dict(item) for item in ledger.get("cohorts", [])]
    if any(item.get("cohort_id") == cohort_id for item in existing):
        raise ValueError(f"cohort already exists in the campaign ledger: {cohort_id}")
    forecast = (
        float(canary["maximum_measured_cell_cost_usd"])
        * cohort_predictions
        * safety_margin
    )
    accounted = sum(float(item["accounted_cost_usd"]) for item in existing)
    requested = float(canary["accounted_cost_usd"]) + forecast
    cap = float(ledger["cap_usd"])
    if accounted + requested > cap:
        raise ValueError(
            f"cohort forecast ${requested:.2f} exceeds remaining campaign budget "
            f"${cap - accounted:.2f}"
        )
    entry = {
        "cohort_id": cohort_id,
        "model": model,
        "status": "admitted",
        "canary_predictions": int(canary["predictions"]),
        "canary_measured_predictions": int(canary["measured_predictions"]),
        "canary_observed_cost_usd": float(canary["observed_cost_usd"]),
        "canary_accounted_cost_usd": float(canary["accounted_cost_usd"]),
        "cohort_predictions": cohort_predictions,
        "safety_margin": safety_margin,
        "forecast_cost_usd": forecast,
        "actual_cost_usd": None,
        "accounted_cost_usd": requested,
    }
    result = dict(ledger)
    result["cohorts"] = [*existing, entry]
    result["accounted_cost_usd"] = accounted + requested
    result["remaining_budget_usd"] = cap - result["accounted_cost_usd"]
    return result


def complete_cohort(
    ledger: Mapping[str, Any], *, cohort_id: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    entries = [dict(item) for item in ledger.get("cohorts", [])]
    matching = [item for item in entries if item.get("cohort_id") == cohort_id]
    if len(matching) != 1:
        raise ValueError("cohort must have exactly one admission record")
    entry = matching[0]
    if len(rows) != int(entry["cohort_predictions"]):
        raise ValueError("completed cohort row count does not match its admission")
    validate_canary(
        rows,
        expected_predictions=int(entry["cohort_predictions"]),
        model=str(entry["model"]),
    )
    costs = [_measured_row_cost(row) for row in rows]
    measured = [value for value in costs if value is not None]
    if not measured:
        raise ValueError("completed cohort has no measured cost")
    actual = sum(measured) + (len(costs) - len(measured)) * max(measured)
    entry["status"] = "completed"
    entry["actual_cost_usd"] = actual
    entry["accounted_cost_usd"] = float(entry["canary_accounted_cost_usd"]) + actual
    cap = float(ledger["cap_usd"])
    accounted = sum(float(item["accounted_cost_usd"]) for item in entries)
    if accounted > cap:
        raise ValueError("completed cohort exceeded the cumulative campaign cap")
    result = dict(ledger)
    result["cohorts"] = entries
    result["accounted_cost_usd"] = accounted
    result["remaining_budget_usd"] = cap - accounted
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    admit = subparsers.add_parser("admit")
    admit.add_argument("--ledger", type=Path, required=True)
    admit.add_argument("--canary-rows", type=Path, required=True)
    admit.add_argument("--cohort-id", required=True)
    admit.add_argument("--model", required=True)
    admit.add_argument("--canary-predictions", type=int, required=True)
    admit.add_argument("--cohort-predictions", type=int, required=True)
    admit.add_argument("--cap-usd", type=float, default=2000.0)
    admit.add_argument("--safety-margin", type=float, default=1.5)
    complete = subparsers.add_parser("complete")
    complete.add_argument("--ledger", type=Path, required=True)
    complete.add_argument("--cohort-id", required=True)
    complete.add_argument("--rows", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "admit":
        ledger = load_ledger(args.ledger, cap_usd=args.cap_usd)
        canary = validate_canary(
            load_rows(args.canary_rows),
            expected_predictions=args.canary_predictions,
            model=args.model,
        )
        result = admit_cohort(
            ledger,
            cohort_id=args.cohort_id,
            model=args.model,
            canary=canary,
            cohort_predictions=args.cohort_predictions,
            safety_margin=args.safety_margin,
        )
    else:
        raw = json.loads(args.ledger.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("campaign ledger must be an object")
        result = complete_cohort(
            raw, cohort_id=args.cohort_id, rows=load_rows(args.rows)
        )
    atomic_write_json(args.ledger, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
