#!/usr/bin/env python3
"""Freeze a blinded, paired trace-audit sample from an immutable preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _load(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _pair_key(cell: Mapping[str, Any]) -> tuple[str, str, int]:
    task_id = cell.get("task_id")
    harness = cell.get("harness")
    attempt = cell.get("trial_index")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("preview cell has no task id")
    if not isinstance(harness, str) or not harness:
        raise ValueError("preview cell has no harness")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("preview cell has an invalid attempt")
    return task_id, harness, attempt


def _partition(task_id: str) -> str:
    if "-dev-" in task_id:
        return "development"
    if "-holdout-" in task_id:
        return "holdout"
    raise ValueError(f"task id has no frozen partition prefix: {task_id}")


def _select_keys(
    pair_keys: Sequence[tuple[str, str, int]],
    *,
    fraction: float,
    seed: str,
    behavior_families: Mapping[str, Sequence[str]] | None = None,
) -> list[tuple[str, str, int]]:
    if not 0 < fraction <= 1:
        raise ValueError("audit fraction must be within (0, 1]")
    by_partition: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for key in pair_keys:
        by_partition[_partition(key[0])].append(key)
    total = max(len(by_partition), math.ceil(len(pair_keys) * fraction))
    targets = {
        partition: math.floor(total * len(keys) / len(pair_keys))
        for partition, keys in by_partition.items()
    }
    for partition in sorted(by_partition):
        if targets[partition] == 0:
            targets[partition] = 1
    while sum(targets.values()) < total:
        partition = max(
            sorted(by_partition),
            key=lambda item: len(by_partition[item]) - targets[item],
        )
        targets[partition] += 1
    while sum(targets.values()) > total:
        partition = max(sorted(by_partition), key=lambda item: targets[item])
        if targets[partition] <= 1:
            raise ValueError("cannot preserve partition coverage")
        targets[partition] -= 1
    selected: list[tuple[str, str, int]] = []
    for partition in sorted(by_partition):
        ranked = sorted(
            by_partition[partition],
            key=lambda key: hashlib.sha256(
                f"{seed}:{partition}:{key[0]}:{key[1]}:{key[2]}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[: targets[partition]])
    selected_set = set(selected)
    for family, task_ids in sorted((behavior_families or {}).items()):
        family_keys = [key for key in pair_keys if key[0] in set(task_ids)]
        if not family_keys:
            raise ValueError(f"behavior family has no preview pair: {family}")
        if any(key in selected_set for key in family_keys):
            continue
        selected_set.add(
            min(
                family_keys,
                key=lambda key: hashlib.sha256(
                    f"{seed}:family:{family}:{key[0]}:{key[1]}:{key[2]}".encode()
                ).hexdigest(),
            )
        )
    return sorted(selected_set)


def freeze(
    preview: Mapping[str, Any],
    *,
    fraction: float,
    behavior_families: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    preview_digest = preview.get("preview_digest")
    matrix = preview.get("matrix")
    if not isinstance(preview_digest, str) or len(preview_digest) != 64:
        raise ValueError("preview digest is invalid")
    if not isinstance(matrix, Mapping):
        raise ValueError("preview matrix is unavailable")
    raw_cells = matrix.get("matrix_cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("preview has no matrix cells")
    pairs: dict[tuple[str, str, int], dict[str, str]] = defaultdict(dict)
    for raw in raw_cells:
        if not isinstance(raw, Mapping) or raw.get("applicable") is not True:
            raise ValueError("trace audit requires applicable matrix cells")
        variant = raw.get("variant_id")
        attempt_id = raw.get("attempt_id")
        if variant not in {"baseline", "candidate"}:
            raise ValueError("trace audit requires baseline and candidate cells")
        if not isinstance(attempt_id, str) or len(attempt_id) != 64:
            raise ValueError("preview cell attempt identity is invalid")
        key = _pair_key(raw)
        if variant in pairs[key]:
            raise ValueError("duplicate treatment cell in an aligned pair")
        pairs[key][str(variant)] = attempt_id
    if any(set(arms) != {"baseline", "candidate"} for arms in pairs.values()):
        raise ValueError("preview contains an incomplete aligned pair")
    selected_keys = _select_keys(
        sorted(pairs),
        fraction=fraction,
        seed=preview_digest,
        behavior_families=behavior_families,
    )
    selected_pairs = []
    for key in selected_keys:
        attempt_ids = sorted(
            pairs[key].values(),
            key=lambda attempt_id: hashlib.sha256(
                f"{preview_digest}:blind:{key[0]}:{key[1]}:{key[2]}:{attempt_id}".encode()
            ).hexdigest(),
        )
        selected_pairs.append(
            {
                "pair_token": hashlib.sha256(
                    f"{preview_digest}:pair:{key[0]}:{key[1]}:{key[2]}".encode()
                ).hexdigest(),
                "task_id": key[0],
                "harness": key[1],
                "attempt": key[2],
                "partition": _partition(key[0]),
                "artifact_a_attempt_id": attempt_ids[0],
                "artifact_b_attempt_id": attempt_ids[1],
            }
        )
    document: dict[str, Any] = {
        "schema_version": 1,
        "kind": "blinded_trace_audit_selection",
        "preview_digest": preview_digest,
        "selection_frozen_before_execution": True,
        "sampling_fraction": fraction,
        "population_pairs": len(pairs),
        "selected_pairs": selected_pairs,
        "selected_attempt_ids": sorted(
            attempt_id
            for pair in selected_pairs
            for attempt_id in (
                pair["artifact_a_attempt_id"],
                pair["artifact_b_attempt_id"],
            )
        ),
        "post_result_additions": {
            "all_discordant_pairs": "required",
            "all_critical_failures": "required",
        },
    }
    document["selection_digest"] = _digest(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preview", type=Path)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument(
        "--preregistration",
        type=Path,
        help="Optional frozen preregistration containing design.behavior_families.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    behavior_families = None
    if args.preregistration is not None:
        preregistration = _load(args.preregistration, "preregistration")
        design = preregistration.get("design")
        if not isinstance(design, Mapping):
            raise ValueError("preregistration design is unavailable")
        raw_families = design.get("behavior_families")
        if not isinstance(raw_families, Mapping):
            raise ValueError("preregistration behavior families are unavailable")
        behavior_families = {
            str(family): tuple(str(task_id) for task_id in task_ids)
            for family, task_ids in raw_families.items()
            if isinstance(task_ids, list)
        }
    document = freeze(
        _load(args.preview, "preview"),
        fraction=args.fraction,
        behavior_families=behavior_families,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selection_digest": document["selection_digest"]}))


if __name__ == "__main__":
    main()
