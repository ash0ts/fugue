from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_PROFILES = ("responses-proxy", "responses-inline", "chat-inline")
_CONTRASTS = (
    ("responses-inline", "responses-proxy", "refactor_topology"),
    ("chat-inline", "responses-inline", "responses_stack_gap"),
)


def analyze_wba_transport_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 2_000,
    seed: str = "wba-transport-ablation-v1",
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    selected = [
        row
        for row in rows
        if row.get("harness") == "wba-responses"
        and row.get("transport_profile") in _PROFILES
    ]
    coordinates: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    duplicates: list[str] = []
    for row in selected:
        task = str(row.get("task_name") or row.get("task_id") or "")
        attempt = int(row.get("trial_index") or 0)
        profile = str(row["transport_profile"])
        key = (task, attempt)
        existing = coordinates.setdefault(key, {})
        if profile in existing:
            duplicates.append(f"{task}:{attempt}:{profile}")
        existing[profile] = row
    incomplete = [
        {
            "task_id": task,
            "attempt": attempt,
            "missing_profiles": sorted(set(_PROFILES) - set(values)),
        }
        for (task, attempt), values in sorted(coordinates.items())
        if set(values) != set(_PROFILES)
    ]
    arm_totals = {
        profile: {
            "passes": sum(
                row.get("pass") is True
                for row in selected
                if row.get("transport_profile") == profile
            ),
            "trials": sum(row.get("transport_profile") == profile for row in selected),
            "provider_or_bridge_errors": sum(
                int(row.get("provider_error_count") or 0)
                + int(row.get("fugue_error_count") or 0)
                for row in selected
                if row.get("transport_profile") == profile
            ),
            "tool_errors": sum(
                int(row.get("transport_tool_errors") or 0)
                for row in selected
                if row.get("transport_profile") == profile
            ),
            "retries": sum(
                int(row.get("transport_retries") or 0)
                for row in selected
                if row.get("transport_profile") == profile
            ),
            "transport_errors": sum(
                int(row.get("transport_errors") or 0)
                for row in selected
                if row.get("transport_profile") == profile
            ),
            "compactions": sum(
                int(row.get("transport_compactions") or 0)
                for row in selected
                if row.get("transport_profile") == profile
            ),
            "orphan_tool_outputs": sum(
                int(row.get("transport_orphan_tool_outputs") or 0)
                for row in selected
                if row.get("transport_profile") == profile
            ),
            "normalization_errors": sum(
                int(row.get("transport_normalization_errors") or 0)
                for row in selected
                if row.get("transport_profile") == profile
            ),
            "stream_anomalies": sum(
                int(row.get("transport_stream_anomalies") or 0)
                for row in selected
                if row.get("transport_profile") == profile
            ),
            "stream_anomaly_kinds": _sum_stream_anomaly_kinds(
                row for row in selected if row.get("transport_profile") == profile
            ),
        }
        for profile in _PROFILES
    }
    contrasts = [
        _contrast(
            coordinates,
            treatment=treatment,
            reference=reference,
            contrast_id=contrast_id,
            bootstrap_samples=bootstrap_samples,
            seed=f"{seed}:{contrast_id}",
        )
        for treatment, reference, contrast_id in _CONTRASTS
    ]
    return {
        "schema_version": 1,
        "analysis_id": "wba-transport-ablation-v1",
        "profiles": list(_PROFILES),
        "complete_grid": not duplicates and not incomplete and bool(coordinates),
        "aligned_coordinates": len(coordinates),
        "duplicate_coordinates": duplicates,
        "incomplete_coordinates": incomplete,
        "arm_totals": arm_totals,
        "contrasts": contrasts,
        "bootstrap_samples": bootstrap_samples,
        "interpretation_guardrails": [
            "A null interval does not establish transport equivalence.",
            "Results apply only to this compatible Fugue harness, locked model, tasks, and attempts.",
            "Task failures are observations; evidence and infrastructure failures invalidate progression.",
        ],
    }


def analyze_wba_transport_v2_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 2_000,
    seed: str = "wba-transport-ablation-v2",
) -> dict[str, Any]:
    """Analyze protocol, operational, and task outcomes without conflating them."""

    report = analyze_wba_transport_rows(
        rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    selected = [
        row
        for row in rows
        if row.get("harness") == "wba-responses"
        and row.get("transport_profile") in _PROFILES
    ]
    for profile in _PROFILES:
        profile_rows = [
            row for row in selected if row.get("transport_profile") == profile
        ]
        report["arm_totals"][profile].update(
            {
                "protocol_applicable": sum(
                    row.get("transport_protocol_applicability") == "applicable"
                    for row in profile_rows
                ),
                "protocol_conformant": sum(
                    row.get("transport_protocol_status") == "conformant"
                    for row in profile_rows
                ),
                "protocol_nonconformant": sum(
                    row.get("transport_protocol_status") == "nonconformant"
                    for row in profile_rows
                ),
                "turn_integrity_errors": sum(
                    int(row.get("transport_turn_integrity_errors") or 0)
                    for row in profile_rows
                ),
                "agent_retries": sum(
                    int(row.get("transport_agent_retries") or 0) for row in profile_rows
                ),
                "agent_errors": sum(
                    int(row.get("transport_agent_errors") or 0) for row in profile_rows
                ),
                "compaction_errors": sum(
                    int(row.get("transport_compaction_errors") or 0)
                    for row in profile_rows
                ),
                "compaction_fallbacks": sum(
                    int(row.get("transport_compaction_fallbacks") or 0)
                    for row in profile_rows
                ),
                "answer_present": sum(
                    _score_component(row, "answer_present") for row in profile_rows
                ),
                "artifact_schema": sum(
                    _score_component(row, "artifact_schema") for row in profile_rows
                ),
                "artifact_facts": sum(
                    _score_component(row, "artifact_facts") for row in profile_rows
                ),
            }
        )
    outcomes = [
        row.get("pass") for row in selected if isinstance(row.get("pass"), bool)
    ]
    report.update(
        {
            "schema_version": 2,
            "analysis_id": "wba-transport-ablation-v2",
            "sections": {
                "protocol_conformance": [
                    "transport_protocol_applicability",
                    "transport_protocol_status",
                    "transport_turn_integrity_errors",
                ],
                "task_outcomes": [
                    "answer_present",
                    "artifact_schema",
                    "artifact_facts",
                    "task_pass",
                ],
                "operations": [
                    "transport_agent_retries",
                    "transport_agent_errors",
                    "transport_compaction_errors",
                    "transport_compaction_fallbacks",
                    "latency",
                    "tokens",
                    "available_cost",
                ],
            },
            "non_discriminating": bool(outcomes) and len(set(outcomes)) == 1,
            "non_discriminating_reason": (
                "All observed deterministic task outcomes are identical; this cohort "
                "does not discriminate among transport profiles."
                if outcomes and len(set(outcomes)) == 1
                else None
            ),
            "interpretation_guardrails": [
                "A non-discriminating cohort does not establish transport equivalence.",
                "Wire conformance, Agent-loop behavior, and task outcomes are separate results.",
                "Results apply only to this compatible Fugue harness, locked model, tasks, and attempts.",
            ],
        }
    )
    return report


def _score_component(row: Mapping[str, Any], name: str) -> int:
    values = row.get("task_score_components") or {}
    return int(isinstance(values, Mapping) and values.get(name) == 1.0)


def _sum_stream_anomaly_kinds(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    totals: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        values = row.get("transport_stream_anomaly_kinds") or {}
        if isinstance(values, Mapping):
            for kind, count in values.items():
                totals[str(kind)] += int(count or 0)
    return dict(sorted(totals.items()))


def _contrast(
    coordinates: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    *,
    treatment: str,
    reference: str,
    contrast_id: str,
    bootstrap_samples: int,
    seed: str,
) -> dict[str, Any]:
    aligned = [
        (key, values)
        for key, values in sorted(coordinates.items())
        if treatment in values
        and reference in values
        and isinstance(values[treatment].get("pass"), bool)
        and isinstance(values[reference].get("pass"), bool)
    ]
    deltas = [
        int(values[treatment]["pass"]) - int(values[reference]["pass"])
        for _key, values in aligned
    ]
    task_deltas: dict[str, list[int]] = defaultdict(list)
    for (task, _attempt), values in aligned:
        task_deltas[task].append(
            int(values[treatment]["pass"]) - int(values[reference]["pass"])
        )
    interval = _cluster_interval(
        task_deltas,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "id": contrast_id,
        "treatment": treatment,
        "reference": reference,
        "aligned_cells": len(aligned),
        "pass_rate_delta": sum(deltas) / len(deltas) if deltas else None,
        "task_cluster_bootstrap_95": interval,
        "discordance": {
            "treatment_only_pass": sum(delta == 1 for delta in deltas),
            "reference_only_pass": sum(delta == -1 for delta in deltas),
            "same_outcome": sum(delta == 0 for delta in deltas),
        },
    }


def _cluster_interval(
    task_deltas: Mapping[str, Sequence[int]],
    *,
    samples: int,
    seed: str,
) -> list[float] | None:
    tasks = sorted(task_deltas)
    if not tasks:
        return None
    generator = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    estimates: list[float] = []
    for _ in range(samples):
        selected = [generator.choice(tasks) for _task in tasks]
        values = [delta for task in selected for delta in task_deltas[task]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return [low, high]
