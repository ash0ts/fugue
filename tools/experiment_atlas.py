#!/usr/bin/env python3
"""Build reviewed, public-safe experiment evidence for the static atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import yaml

from fugue.bench.reproducibility import verify_snapshot
from fugue.bench.scoring import _paired_delta_interval
from fugue.model_plane import (
    ModelRoute,
    Provider,
    ToolResultModality,
    model_route_identity,
    resolve_harness_model_route,
)

PUBLIC_EXPERIMENT_SCHEMA_VERSION = 1
PUBLIC_EXPERIMENT_V2_SCHEMA_VERSION = 2
EXPERIMENT_INDEX_SCHEMA_VERSION = 1
INVENTORY_SCHEMA_VERSION = 1
EVIDENCE_TIERS = {
    "confirmed": 1,
    "directional": 2,
    "baseline": 3,
    "contract": 4,
    "active": 5,
    "blocked": 5,
}
COMPLETE_TIERS = {"confirmed", "directional", "baseline", "contract"}
ALLOWED_URL_HOSTS = {
    "wandb.ai",
    "app.wandb.ai",
    "github.com",
    "docs.wandb.ai",
    "platform.claude.com",
}
ALLOWED_MODEL_UPSTREAM_HOSTS = {
    "api.inference.wandb.ai",
    "api.openai.com",
    "api.anthropic.com",
}
EDITORIAL_FIELDS = {
    "schema_version",
    "id",
    "title",
    "summary",
    "question",
    "hypothesis",
    "why_it_matters",
    "task_selection",
    "evidence_tier",
    "decision_value",
    "status",
    "matrix",
    "provenance",
    "links",
    "caveats",
    "findings",
}
V2_EDITORIAL_FIELDS = EDITORIAL_FIELDS | {
    "study_kind",
    "publication_level",
    "primary_outcome",
    "ledgers",
    "metrics",
    "groups",
    "cells",
}
MATRIX_FIELDS = {
    "experiment_id",
    "workload_id",
    "expected_predictions",
    "attempts",
    "models",
    "harnesses",
    "treatments",
    "tasks",
    "cohorts",
}
COHORT_FIELDS = {
    "id",
    "label",
    "models",
    "harnesses",
    "treatments",
    "tasks",
    "expected_predictions",
}
PROVENANCE_FIELDS = {
    "source_commit",
    "source_url",
    "dataset_id",
    "dataset_digest",
    "snapshot_digest",
    "run_ids",
}
LINK_FIELDS = {"project", "evaluations"}

PUBLIC_CELL_FIELDS = {
    "prediction_id",
    "run_id",
    "candidate_id",
    "comparison_example_id",
    "trial_index",
    "execution_kind",
    "workload_id",
    "task_id",
    "harness",
    "treatment",
    "provider",
    "model",
    "wire_protocol",
    "endpoint_kind",
    "upstream_host",
    "route_evidence",
    "status",
    "pass",
    "reward",
    "wall_time_sec",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "tool_calls",
    "turns",
    "recoverable_errors",
    "refusals",
    "provider_errors",
    "harness_errors",
    "context_registered",
    "context_invoked",
    "context_invocation_count",
    "recall_at_10",
    "mrr",
    "agent_link",
}
V2_PUBLIC_CELL_FIELDS = {
    "cell_id",
    "run_id",
    "candidate_id",
    "task_id",
    "harness",
    "treatment",
    "attempt",
    "status",
    "outcome",
    "outcome_label",
    "cost_usd",
    "evidence_link",
}
V2_METRIC_FIELDS = {
    "expected_cells",
    "published_cells",
    "outcome_observed_cells",
    "outcome_successes",
    "outcome_rate",
    "total_cost_usd",
    "aligned_pairs",
    "improved_pairs",
    "regressed_pairs",
    "unchanged_pairs",
    "critical_failures",
}
PRIMARY_OUTCOME_FIELDS = {"id", "label", "success_label", "failure_label"}
LEDGER_NAMES = {
    "infrastructure",
    "deterministic",
    "authored_judge",
    "mechanism",
    "evidence_integrity",
    "decision",
}
LEDGER_STATUSES = {
    "passed",
    "failed",
    "mixed",
    "advisory",
    "unavailable",
    "hold",
    "not_applicable",
}
INVENTORY_ENTRY_FIELDS = {
    "id",
    "label",
    "category",
    "lifecycle_state",
    "publication_level",
    "run_ids",
    "study_ids",
    "source_reference",
    "claim_boundary",
}
FORBIDDEN_KEY = re.compile(
    r"(?:prompt|response|reasoning|message|tool_(?:argument|result|output)|"
    r"gold|expected_path|environment|env_|secret|credential|api_?key|exception)",
    re.IGNORECASE,
)
SECRET_VALUE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|api[_ -]?key\s*[:=]|bearer\s+[A-Za-z0-9._-]{12,})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PublicExperimentV1:
    schema_version: int
    id: str
    title: str
    summary: str
    question: str
    hypothesis: str
    why_it_matters: str
    task_selection: str
    evidence_tier: str
    decision_value: int
    status: str
    matrix: dict[str, Any]
    provenance: dict[str, Any]
    links: dict[str, Any]
    findings: tuple[str, ...]
    caveats: tuple[str, ...]
    metrics: dict[str, Any]
    groups: tuple[dict[str, Any], ...]
    cells: tuple[dict[str, Any], ...]
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicExperimentV2:
    schema_version: int
    id: str
    title: str
    summary: str
    question: str
    hypothesis: str
    why_it_matters: str
    task_selection: str
    evidence_tier: str
    decision_value: int
    status: str
    study_kind: str
    publication_level: str
    primary_outcome: dict[str, str]
    ledgers: dict[str, dict[str, str]]
    matrix: dict[str, Any]
    provenance: dict[str, Any]
    links: dict[str, Any]
    findings: tuple[str, ...]
    caveats: tuple[str, ...]
    metrics: dict[str, Any]
    groups: tuple[dict[str, Any], ...]
    cells: tuple[dict[str, Any], ...]
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PublicExperiment = PublicExperimentV1 | PublicExperimentV2


@dataclass(frozen=True)
class ExperimentIndexV1:
    schema_version: int
    experiments: tuple[dict[str, Any], ...]
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_editorial(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("editorial record must be a mapping")
    schema_version = raw.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError("editorial schema_version must be 1 or 2")
    _exact_fields(
        raw,
        EDITORIAL_FIELDS if schema_version == 1 else V2_EDITORIAL_FIELDS,
        "editorial record",
    )
    for field in (
        "id",
        "title",
        "summary",
        "question",
        "hypothesis",
        "why_it_matters",
        "task_selection",
        "evidence_tier",
        "status",
    ):
        if not isinstance(raw.get(field), str) or not str(raw[field]).strip():
            raise ValueError(f"editorial {field} must be non-empty text")
    tier = str(raw["evidence_tier"])
    if tier not in EVIDENCE_TIERS:
        raise ValueError(f"unsupported evidence tier: {tier}")
    value = raw.get("decision_value")
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
        raise ValueError("decision_value must be an integer from 0 through 100")
    matrix = _mapping(raw.get("matrix"), "matrix")
    provenance = _mapping(raw.get("provenance"), "provenance")
    links = _mapping(raw.get("links"), "links")
    _exact_fields(matrix, MATRIX_FIELDS, "matrix")
    _exact_fields(provenance, PROVENANCE_FIELDS, "provenance")
    _exact_fields(links, LINK_FIELDS, "links")
    cohorts = matrix.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise ValueError("matrix.cohorts must be a non-empty list")
    cohort_ids: set[str] = set()
    for cohort in cohorts:
        value = _mapping(cohort, "matrix cohort")
        _exact_fields(value, COHORT_FIELDS, "matrix cohort")
        cohort_id = str(value.get("id") or "")
        if not cohort_id or cohort_id in cohort_ids:
            raise ValueError("matrix cohort IDs must be non-empty and unique")
        cohort_ids.add(cohort_id)
        for field in ("models", "harnesses", "treatments", "tasks"):
            _string_list(value.get(field), f"matrix cohort {field}")
        if int(value.get("expected_predictions") or 0) < 1:
            raise ValueError("matrix cohort expected_predictions must be positive")
    if sum(int(item["expected_predictions"]) for item in cohorts) != int(
        matrix["expected_predictions"]
    ):
        raise ValueError("matrix cohort prediction counts do not match the matrix")
    for url in _all_urls(links) | _all_urls(provenance):
        _validate_url(url)
    if schema_version == 2:
        _validate_v2_contract(raw, editorial=True)
    _reject_sensitive(raw)
    return raw


def _validate_v2_contract(value: Mapping[str, Any], *, editorial: bool) -> None:
    if value.get("study_kind") not in {
        "benchmark",
        "comparison",
        "safety",
        "contract",
    }:
        raise ValueError("V2 study_kind is unsupported")
    publication_level = value.get("publication_level")
    if publication_level not in {"full", "summary"}:
        raise ValueError("V2 publication_level must be full or summary")
    outcome = _mapping(value.get("primary_outcome"), "V2 primary outcome")
    _exact_fields(outcome, PRIMARY_OUTCOME_FIELDS, "V2 primary outcome")
    if any(not isinstance(outcome.get(field), str) or not outcome[field] for field in PRIMARY_OUTCOME_FIELDS):
        raise ValueError("V2 primary outcome fields must be non-empty text")
    ledgers = _mapping(value.get("ledgers"), "V2 ledgers")
    _exact_fields(ledgers, LEDGER_NAMES, "V2 ledgers")
    for name, raw_ledger in ledgers.items():
        ledger = _mapping(raw_ledger, f"V2 {name} ledger")
        _exact_fields(ledger, {"status", "summary"}, f"V2 {name} ledger")
        if ledger.get("status") not in LEDGER_STATUSES:
            raise ValueError(f"V2 {name} ledger status is unsupported")
        if not isinstance(ledger.get("summary"), str) or not ledger["summary"]:
            raise ValueError(f"V2 {name} ledger summary must be non-empty")
    metrics = _mapping(value.get("metrics"), "V2 metrics")
    _exact_fields(metrics, V2_METRIC_FIELDS, "V2 metrics")
    expected = int(metrics.get("expected_cells") or 0)
    published = int(metrics.get("published_cells") or 0)
    observed = int(metrics.get("outcome_observed_cells") or 0)
    successes = int(metrics.get("outcome_successes") or 0)
    if expected < 1 or not 0 <= successes <= observed <= published <= expected:
        raise ValueError("V2 metrics contain an invalid denominator")
    rate = metrics.get("outcome_rate")
    expected_rate = successes / observed if observed else None
    if rate != expected_rate:
        raise ValueError("V2 outcome rate does not match its denominator")
    cells = value.get("cells")
    if not isinstance(cells, (list, tuple)):
        raise ValueError("V2 cells must be a list")
    if publication_level == "summary" and cells:
        raise ValueError("summary-only evidence cannot be promoted to task-level cells")
    if publication_level == "full" and len(cells) != published:
        raise ValueError("full V2 evidence must publish every declared cell")
    for cell in cells:
        cell_value = _mapping(cell, "V2 public cell")
        _exact_fields(cell_value, V2_PUBLIC_CELL_FIELDS, "V2 public cell")
        if cell_value.get("outcome") not in {True, False, None}:
            raise ValueError("V2 cell outcome must be true, false, or unavailable")
        evidence_link = cell_value.get("evidence_link")
        if evidence_link:
            _validate_url(str(evidence_link))
    if publication_level == "full":
        cell_outcomes = [cell["outcome"] for cell in cells if cell["outcome"] is not None]
        if len(cell_outcomes) != observed or sum(item is True for item in cell_outcomes) != successes:
            raise ValueError("V2 metrics do not match published cells")
    groups = value.get("groups")
    if not isinstance(groups, (list, tuple)):
        raise ValueError("V2 groups must be a list")
    if editorial and value.get("schema_version") != 2:
        raise ValueError("V2 editorial contract requires schema 2")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL row {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        rows.append(row)
    return rows


def build_public_experiment(
    editorial: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    run_summaries: Sequence[Mapping[str, Any]] = (),
) -> PublicExperimentV1:
    tier = str(editorial["evidence_tier"])
    evaluation_links = _evaluation_links(editorial, rows, run_summaries)
    safe_cells = tuple(_public_cell(row, evaluation_links) for row in rows)
    prediction_ids = [str(cell["prediction_id"]) for cell in safe_cells]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("public experiment contains duplicate prediction IDs")
    expected = int(_mapping(editorial["matrix"], "matrix")["expected_predictions"])
    if tier in COMPLETE_TIERS and len(safe_cells) != expected:
        raise ValueError(
            f"{tier} evidence requires {expected} predictions; found {len(safe_cells)}"
        )
    if tier in {"active", "blocked"} and safe_cells:
        raise ValueError(f"{tier} evidence cannot publish partial result rows")
    _validate_compatible_cohort(editorial, safe_cells, complete=tier in COMPLETE_TIERS)
    metrics = _metrics(safe_cells, expected)
    metrics["paired_bootstrap"] = (
        _confirmed_intervals(safe_cells, str(editorial["id"]))
        if tier == "confirmed"
        else None
    )
    groups = tuple(_group_metrics(safe_cells))
    body = {
        "schema_version": PUBLIC_EXPERIMENT_SCHEMA_VERSION,
        "id": str(editorial["id"]),
        "title": str(editorial["title"]),
        "summary": str(editorial["summary"]),
        "question": str(editorial["question"]),
        "hypothesis": str(editorial["hypothesis"]),
        "why_it_matters": str(editorial["why_it_matters"]),
        "task_selection": str(editorial["task_selection"]),
        "evidence_tier": tier,
        "decision_value": int(editorial["decision_value"]),
        "status": str(editorial["status"]),
        "matrix": dict(editorial["matrix"]),
        "provenance": dict(editorial["provenance"]),
        "links": {
            "project": editorial["links"]["project"],
            "evaluations": sorted(set(evaluation_links.values())),
        },
        "findings": tuple(_string_list(editorial.get("findings"), "findings")),
        "caveats": tuple(_string_list(editorial.get("caveats"), "caveats")),
        "metrics": metrics,
        "groups": groups,
        "cells": safe_cells,
    }
    public = PublicExperimentV1(
        **body,
        content_sha256=_digest(body),
    )
    validate_public_experiment(public.to_dict())
    return public


def build_public_experiment_v2(
    editorial: Mapping[str, Any],
) -> PublicExperimentV2:
    """Build a public V2 record from an already reviewed public-safe projection."""
    if editorial.get("schema_version") != PUBLIC_EXPERIMENT_V2_SCHEMA_VERSION:
        raise ValueError("PublicExperimentV2 requires editorial schema 2")
    _validate_v2_contract(editorial, editorial=True)
    body = {
        key: (
            tuple(value)
            if key in {"findings", "caveats", "groups", "cells"}
            else value
        )
        for key, value in editorial.items()
    }
    public = PublicExperimentV2(
        **body,
        content_sha256=_digest(body),
    )
    validate_public_experiment(public.to_dict())
    return public


def build_index(experiments: Iterable[PublicExperiment]) -> ExperimentIndexV1:
    ordered = sorted(
        experiments,
        key=lambda item: (
            EVIDENCE_TIERS[item.evidence_tier],
            -item.decision_value,
            item.title.casefold(),
        ),
    )
    records = tuple(
        {
            "id": item.id,
            "title": item.title,
            "summary": item.summary,
            "evidence_tier": item.evidence_tier,
            "decision_value": item.decision_value,
            "status": item.status,
            "metrics": item.metrics,
            "models": item.matrix.get("models", []),
            "harnesses": item.matrix.get("harnesses", []),
            "treatments": item.matrix.get("treatments", []),
            "content_sha256": item.content_sha256,
        }
        for item in ordered
    )
    body = {
        "schema_version": EXPERIMENT_INDEX_SCHEMA_VERSION,
        "experiments": records,
    }
    index = ExperimentIndexV1(**body, content_sha256=_digest(body))
    validate_experiment_index(index.to_dict())
    return index


def write_publication(
    editorial_paths: Sequence[Path],
    row_paths: Mapping[str, Path],
    run_summary_paths: Mapping[str, Sequence[Path]],
    snapshot_paths: Mapping[str, Sequence[Path]],
    output: Path,
    *,
    repo_root: Path,
) -> ExperimentIndexV1:
    output.mkdir(parents=True, exist_ok=True)
    experiments: list[PublicExperiment] = []
    for path in sorted(editorial_paths):
        editorial = load_editorial(path)
        experiment_id = str(editorial["id"])
        rows_path = row_paths.get(experiment_id)
        rows = load_rows(rows_path) if rows_path else []
        summaries = [
            json.loads(summary.read_text(encoding="utf-8"))
            for summary in run_summary_paths.get(experiment_id, ())
        ]
        snapshots = [
            json.loads(snapshot.read_text(encoding="utf-8"))
            for snapshot in snapshot_paths.get(experiment_id, ())
        ]
        if rows:
            editorial = {
                **editorial,
                "provenance": _validated_provenance(
                    editorial,
                    rows,
                    snapshots,
                    repo_root=repo_root,
                ),
            }
            rows = _attach_snapshot_route_receipts(rows, snapshots)
        elif snapshots:
            raise ValueError(
                "planned or blocked experiments cannot attach run snapshots"
            )
        if editorial["schema_version"] == 2:
            if rows or summaries or snapshots:
                raise ValueError(
                    "V2 editorial consumes reviewed public projections, not private inputs"
                )
            public = build_public_experiment_v2(editorial)
        elif not rows and (output / "experiments" / f"{experiment_id}.json").exists():
            existing = json.loads(
                (output / "experiments" / f"{experiment_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            validate_public_experiment(existing)
            _validate_editorial_projection(editorial, existing)
            public = PublicExperimentV1(**existing)
        else:
            public = build_public_experiment(editorial, rows, summaries)
        _write_json(output / "experiments" / f"{experiment_id}.json", public.to_dict())
        experiments.append(public)
    index = build_index(experiments)
    _write_json(output / "index.json", index.to_dict())
    _write_or_validate_inventory(
        editorial_paths[0].parent,
        output,
        {item.id for item in experiments},
        write=True,
    )
    return index


def validate_publication(
    editorial_paths: Sequence[Path], output: Path
) -> ExperimentIndexV1:
    editorials = {
        str(value["id"]): value
        for value in (load_editorial(path) for path in sorted(editorial_paths))
    }
    experiment_dir = output / "experiments"
    published_paths = sorted(experiment_dir.glob("*.json"))
    if {path.stem for path in published_paths} != set(editorials):
        raise ValueError(
            "reviewed public snapshots do not match the editorial registry"
        )
    experiments: list[PublicExperiment] = []
    for path in published_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("reviewed public snapshot must be an object")
        validate_public_experiment(value)
        _validate_editorial_projection(editorials[path.stem], value)
        if value.get("schema_version") == 1:
            experiments.append(PublicExperimentV1(**value))
        else:
            experiments.append(PublicExperimentV2(**value))
    expected = build_index(experiments)
    index_value = json.loads((output / "index.json").read_text(encoding="utf-8"))
    if not isinstance(index_value, dict):
        raise ValueError("reviewed public index must be an object")
    validate_experiment_index(index_value)
    if _canonical_json(index_value) != _canonical_json(expected.to_dict()):
        raise ValueError(
            "reviewed public index does not match its experiment snapshots"
        )
    _write_or_validate_inventory(
        editorial_paths[0].parent,
        output,
        set(editorials),
        write=False,
    )
    return expected


def _validate_editorial_projection(
    editorial: Mapping[str, Any], public: Mapping[str, Any]
) -> None:
    fields: tuple[str, ...] = (
        "schema_version",
        "id",
        "title",
        "summary",
        "question",
        "hypothesis",
        "why_it_matters",
        "task_selection",
        "evidence_tier",
        "decision_value",
        "status",
        "matrix",
        "provenance",
        "findings",
        "caveats",
    )
    if editorial.get("schema_version") == 2:
        fields += (
            "study_kind",
            "publication_level",
            "primary_outcome",
            "ledgers",
            "metrics",
            "groups",
            "cells",
        )
    for field in fields:
        if _canonical_json(editorial[field]) != _canonical_json(public[field]):
            raise ValueError(
                f"reviewed public snapshot does not match editorial {field}"
            )
    links = _mapping(public.get("links"), "public links")
    if links.get("project") != editorial["links"]["project"]:
        raise ValueError(
            "reviewed public snapshot does not match editorial project link"
        )


def _validated_provenance(
    editorial: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if not snapshots or any(not verify_snapshot(snapshot) for snapshot in snapshots):
        raise ValueError(
            "complete public evidence requires valid immutable run snapshots"
        )
    declared = _mapping(editorial["provenance"], "provenance")
    expected_runs = set(_string_list(declared["run_ids"], "run_ids"))
    row_runs = {str(row.get("run_id") or "") for row in rows}
    snapshot_runs = {str(snapshot.get("run_id") or "") for snapshot in snapshots}
    if not row_runs or row_runs != expected_runs or snapshot_runs != expected_runs:
        raise ValueError("rows, snapshots, and declared run provenance do not match")

    source_commits: set[str] = set()
    manifests: set[str] = set()
    snapshot_digests: dict[str, str] = {}
    planned: dict[str, set[tuple[str, str, int, str, str, str]]] = {}
    experiment_ids: set[str] = set()
    workload_id = str(_mapping(editorial["matrix"], "matrix")["workload_id"])
    for snapshot in snapshots:
        run_id = str(snapshot["run_id"])
        snapshot_digests[run_id] = str(snapshot["snapshot_sha256"])
        request = _mapping(snapshot.get("request"), "snapshot request")
        manifests.add(_resolved_workload_manifest(snapshot, workload_id))
        experiment_ids.add(str(request.get("experiment_id") or ""))
        runtime = _mapping(snapshot.get("runtime"), "snapshot runtime")
        executions = _mapping(runtime.get("executions"), "snapshot executions")
        for execution in executions.values():
            source = _mapping(
                _mapping(execution, "snapshot execution").get("fugue_source"),
                "snapshot Fugue source",
            )
            if source.get("kind") != "git" or source.get("dirty") is not False:
                raise ValueError(
                    "public evidence requires a clean tracked Fugue source"
                )
            source_commits.add(str(source.get("commit") or ""))
        coordinates: set[tuple[str, str, int, str, str, str]] = set()
        for cell in snapshot.get("planned_matrix") or []:
            value = _mapping(cell, "snapshot planned cell")
            if not value.get("applicable", True):
                continue
            coordinates.add(
                (
                    str(value.get("candidate_id") or ""),
                    str(value.get("comparison_example_id") or ""),
                    int(value.get("trial_index") or 0),
                    str(value.get("execution_kind") or ""),
                    str(value.get("workload_id") or ""),
                    str(value.get("task_id") or ""),
                )
            )
        planned[run_id] = coordinates

    if len(source_commits) != 1 or not next(iter(source_commits), ""):
        raise ValueError("public runs do not share one immutable Fugue source commit")
    if len(manifests) != 1 or not next(iter(manifests), ""):
        raise ValueError("public runs do not share one resolved workload manifest")
    if experiment_ids != {
        str(_mapping(editorial["matrix"], "matrix")["experiment_id"])
    }:
        raise ValueError("public runs do not match the editorial experiment")
    observed_agent_coordinates: set[tuple[str, str, int, str, str, str]] = set()
    for run_id in expected_runs:
        for row in (row for row in rows if str(row.get("run_id") or "") == run_id):
            if row.get("execution_kind") != "agent":
                continue
            coordinate = (
                str(row.get("candidate_id") or ""),
                str(row.get("comparison_example_id") or ""),
                int(row.get("trial_index") or 0),
                str(row.get("execution_kind") or ""),
                str(row.get("workload_id") or ""),
                str(row.get("task_id") or row.get("task_name") or ""),
            )
            if coordinate not in planned[run_id]:
                raise ValueError("normalized row is outside its immutable run snapshot")
            observed_agent_coordinates.add(coordinate)
    planned_agent_coordinates = {
        coordinate
        for coordinates in planned.values()
        for coordinate in coordinates
        if coordinate[3] == "agent"
    }
    if observed_agent_coordinates != planned_agent_coordinates:
        raise ValueError(
            "normalized Agent rows do not cover the frozen matrix coordinates"
        )

    manifest = next(iter(manifests))
    manifest_path = (repo_root / manifest).resolve()
    try:
        manifest_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError("dataset manifest escapes the repository") from exc
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest is unavailable: {manifest}")
    dataset_id = manifest_path.stem
    dataset_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    source_commit = next(iter(source_commits))
    snapshot_digest = (
        next(iter(snapshot_digests.values()))
        if len(snapshot_digests) == 1
        else _digest({"snapshots": dict(sorted(snapshot_digests.items()))})
    )
    derived = {
        "source_commit": source_commit,
        "source_url": f"https://github.com/ash0ts/fugue/commit/{source_commit}",
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "snapshot_digest": snapshot_digest,
        "run_ids": sorted(expected_runs),
    }
    for field in PROVENANCE_FIELDS:
        if declared.get(field) != derived[field]:
            raise ValueError(
                f"editorial provenance does not match run evidence: {field}"
            )
    return derived


def _resolved_workload_manifest(snapshot: Mapping[str, Any], workload_id: str) -> str:
    experiment = _mapping(snapshot.get("experiment"), "snapshot resolved experiment")
    matches = [
        _mapping(item, "snapshot workload")
        for item in experiment.get("workloads") or []
        if _mapping(item, "snapshot workload").get("id") == workload_id
    ]
    if len(matches) != 1 or not matches[0].get("manifest"):
        raise ValueError("public evidence requires one resolved workload manifest")
    return str(matches[0]["manifest"])


def _attach_snapshot_route_receipts(
    rows: Sequence[Mapping[str, Any]], snapshots: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    receipts: dict[tuple[str, str], tuple[dict[str, object], str, bool]] = {}
    for snapshot in snapshots:
        run_id = str(snapshot.get("run_id") or "")
        candidate_runtime = _mapping(
            snapshot.get("candidate_runtime"), "snapshot candidate runtime"
        )
        bridge = _mapping(snapshot.get("runtime"), "snapshot runtime").get("bridge")
        for candidate_id, raw_runtime in candidate_runtime.items():
            runtime = _mapping(raw_runtime, "snapshot candidate runtime record")
            model_route = _model_route_from_snapshot(runtime.get("model_route"))
            derived = resolve_harness_model_route(
                model_route, str(runtime.get("harness") or "")
            )
            locked = runtime.get("model_transport")
            if (
                locked is not None
                and _mapping(locked, "snapshot model transport") != derived
            ):
                raise ValueError(
                    "snapshot model transport differs from its canonical route"
                )
            evidence = "snapshot_attested" if locked is not None else "configured_only"
            bridge_locked = not derived["bridge_required"] or _bridge_attests_route(
                bridge, model_route
            )
            if not bridge_locked:
                evidence = "configured_only"
            receipts[(run_id, str(candidate_id))] = (
                derived,
                evidence,
                bridge_locked,
            )

    enriched: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("run_id") or ""), str(row.get("candidate_id") or ""))
        if key not in receipts:
            raise ValueError(
                "normalized row has no model route in its immutable snapshot"
            )
        snapshot_receipt, evidence, bridge_locked = receipts[key]
        observed = row.get("model_transport")
        if observed is not None:
            if _mapping(observed, "runtime model transport") != snapshot_receipt:
                raise ValueError("runtime model transport differs from the snapshot")
            if bridge_locked:
                evidence = "runtime_attested"
        enriched.append(
            {
                **dict(row),
                "model_transport": {**snapshot_receipt, "evidence": evidence},
            }
        )
    return enriched


def _model_route_from_snapshot(value: Any) -> ModelRoute:
    route = _mapping(value, "snapshot model route")
    provider = str(route.get("provider") or "")
    if provider not in {"wandb", "openai", "anthropic"}:
        raise ValueError("snapshot model route has an unsupported provider")
    model_id = str(route.get("model_id") or "")
    display_model = str(route.get("display_model") or "")
    api_key_env = str(route.get("api_key_env") or "")
    litellm_model = str(route.get("litellm_model") or "")
    modalities = tuple(route.get("tool_result_modalities") or ())
    expected_key = {
        "wandb": "WANDB_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[provider]
    if (
        not model_id
        or display_model != f"{provider}/{model_id}"
        or api_key_env != expected_key
        or not litellm_model
        or not modalities
        or any(item not in {"text", "image"} for item in modalities)
    ):
        raise ValueError("snapshot model route is incomplete or inconsistent")
    return ModelRoute(
        provider=cast(Provider, provider),
        model_id=model_id,
        display_model=display_model,
        api_key_env=api_key_env,
        chat_base_url=_optional_text(route.get("chat_base_url")),
        responses_base_url=_optional_text(route.get("responses_base_url")),
        messages_base_url=_optional_text(route.get("messages_base_url")),
        litellm_model=litellm_model,
        tool_result_modalities=cast(tuple[ToolResultModality, ...], modalities),
    )


def _bridge_attests_route(value: Any, route: ModelRoute) -> bool:
    if value is None:
        return False
    bridge = _mapping(value, "snapshot bridge runtime")
    required = {
        "schema_version",
        "image",
        "config_sha256",
        "target_route",
        "resolved_image_id",
    }
    if set(bridge) != required or bridge.get("schema_version") != 1:
        raise ValueError("snapshot bridge runtime does not match schema 1")
    image = str(bridge.get("image") or "")
    resolved_image = str(bridge.get("resolved_image_id") or "")
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", resolved_image
    ):
        raise ValueError("snapshot bridge runtime is not image-digest pinned")
    if not re.fullmatch(r"[0-9a-f]{64}", str(bridge.get("config_sha256") or "")):
        raise ValueError("snapshot bridge config is not a SHA-256 digest")
    if _mapping(
        bridge.get("target_route"), "snapshot bridge target route"
    ) != model_route_identity(route):
        raise ValueError(
            "snapshot bridge target differs from the candidate model route"
        )
    return True


def _public_cell(
    row: Mapping[str, Any], evaluation_links: Mapping[str, str]
) -> dict[str, Any]:
    if row.get("schema_version") != 1 or row.get("prediction_schema_version") != 1:
        raise ValueError("public rows require canonical prediction schema 1")
    if row.get("record_type") != "trial":
        raise ValueError("public experiment rows must be normalized trial records")
    required = (
        "prediction_id",
        "run_id",
        "candidate_id",
        "comparison_example_id",
        "trial_index",
        "execution_kind",
        "harness",
        "model",
    )
    if any(row.get(field) in (None, "") for field in required):
        raise ValueError("normalized row is missing canonical public identity")
    link = _verified_agent_link(row, evaluation_links)
    transport = _mapping(row.get("model_transport") or {}, "model transport")
    cell = {
        "prediction_id": str(row["prediction_id"]),
        "run_id": str(row["run_id"]),
        "candidate_id": str(row["candidate_id"])[:12],
        "comparison_example_id": str(row["comparison_example_id"]),
        "trial_index": int(row["trial_index"]),
        "execution_kind": str(row["execution_kind"]),
        "workload_id": str(row.get("workload_id") or ""),
        "task_id": str(row.get("task_id") or row.get("task_name") or ""),
        "harness": str(row["harness"]),
        "treatment": str(
            row.get("variant_id") or row.get("context_system_id") or "none"
        ),
        "provider": str(row.get("model_provider") or row.get("provider") or ""),
        "model": str(row["model"]),
        "wire_protocol": _optional_text(transport.get("wire_protocol")),
        "endpoint_kind": _optional_text(transport.get("endpoint_kind")),
        "upstream_host": _optional_text(transport.get("upstream_host")),
        "route_evidence": _optional_text(transport.get("evidence")),
        "status": str(row.get("status") or "unknown"),
        "pass": _optional_bool(row.get("pass")),
        "reward": _optional_number(row.get("reward")),
        "wall_time_sec": _optional_number(row.get("wall_time_sec")),
        "cost_usd": _public_cost(row),
        "input_tokens": _optional_int(row.get("n_input_tokens")),
        "output_tokens": _optional_int(row.get("n_output_tokens")),
        "tool_calls": _optional_int(row.get("weave_tool_call_count")),
        "turns": _optional_int(row.get("weave_turn_count")),
        "recoverable_errors": int(row.get("recoverable_error_count") or 0),
        "refusals": int(row.get("refusal_count") or 0),
        "provider_errors": int(row.get("provider_error_count") or 0),
        "harness_errors": int(
            row.get("harness_error_count")
            or row.get("harness_adapter_error_count")
            or 0
        ),
        "context_registered": _optional_bool(row.get("context_registered")),
        "context_invoked": _optional_bool(row.get("context_invoked")),
        "context_invocation_count": _optional_int(row.get("context_invocation_count")),
        "recall_at_10": _optional_number(row.get("recall_at_10")),
        "mrr": _optional_number(row.get("mrr")),
        "agent_link": link,
    }
    assert set(cell) == PUBLIC_CELL_FIELDS
    _validate_public_transport(cell)
    _reject_sensitive(cell)
    return cell


def _validate_public_transport(cell: Mapping[str, Any]) -> None:
    fields = (
        cell.get("wire_protocol"),
        cell.get("endpoint_kind"),
        cell.get("upstream_host"),
        cell.get("route_evidence"),
    )
    if all(value is None for value in fields):
        return
    if any(value is None for value in fields):
        raise ValueError("public model transport must be complete or unavailable")
    if cell["wire_protocol"] not in {"chat_completions", "messages", "responses"}:
        raise ValueError("public model transport has an unsupported wire protocol")
    if cell["endpoint_kind"] not in {"provider_direct", "fugue_bridge"}:
        raise ValueError("public model transport has an unsupported endpoint kind")
    if cell["upstream_host"] not in ALLOWED_MODEL_UPSTREAM_HOSTS:
        raise ValueError("public model transport has an unapproved upstream host")
    if cell["route_evidence"] not in {
        "configured_only",
        "snapshot_attested",
        "runtime_attested",
    }:
        raise ValueError("public model transport has an unsupported evidence level")


def _verified_agent_link(
    row: Mapping[str, Any], evaluation_links: Mapping[str, str]
) -> str | None:
    if row.get("execution_kind") != "agent":
        return None
    status = str(row.get("trace_link_status") or row.get("agent_link_status") or "")
    if status not in {"verified", "linked", "exact"}:
        raise ValueError("agent prediction does not have a verified trace link")
    candidate_id = str(row["candidate_id"])
    link = evaluation_links.get(candidate_id)
    if not link:
        raise ValueError("verified agent prediction is missing its evaluation link")
    return link


def _evaluation_links(
    editorial: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if editorial["links"]["evaluations"]:
        raise ValueError("evaluation links are derived from run summaries")
    if not rows:
        return {}
    expected_runs = set(_string_list(editorial["provenance"]["run_ids"], "run_ids"))
    observed_runs = {str(row.get("run_id") or "") for row in rows}
    summary_runs = {str(summary.get("run_id") or "") for summary in summaries}
    if (
        not observed_runs
        or observed_runs - expected_runs
        or observed_runs - summary_runs
    ):
        raise ValueError("public rows are not covered by declared run summaries")
    links: dict[str, str] = {}
    for summary in summaries:
        if str(summary.get("run_id") or "") not in expected_runs:
            raise ValueError("run summary is outside declared provenance")
        for evaluation in summary.get("evaluation_runs") or []:
            if not evaluation.get("active", True):
                continue
            agent_predictions = int(evaluation.get("agent_predictions") or 0)
            linked = int(evaluation.get("linked_agent_predictions") or 0)
            if agent_predictions < 1 or linked != agent_predictions:
                continue
            if evaluation.get("linking_failures"):
                continue
            candidate_id = str(evaluation.get("candidate_id") or "")
            url = str(evaluation.get("url") or "")
            if not candidate_id or not url:
                continue
            _validate_url(url)
            if candidate_id in links and links[candidate_id] != url:
                raise ValueError("candidate has conflicting evaluation links")
            links[candidate_id] = url
    agent_candidates = {
        str(row.get("candidate_id") or "")
        for row in rows
        if row.get("execution_kind") == "agent"
    }
    if agent_candidates - set(links):
        raise ValueError("run summaries do not verify every Agent candidate link")
    return links


def validate_public_experiment(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") == PUBLIC_EXPERIMENT_V2_SCHEMA_VERSION:
        _validate_public_experiment_v2(value)
        return
    allowed = {field.name for field in PublicExperimentV1.__dataclass_fields__.values()}
    _exact_fields(value, allowed, "public experiment")
    if value.get("schema_version") != PUBLIC_EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("public experiment schema_version must be 1")
    cells = value.get("cells")
    if not isinstance(cells, (list, tuple)):
        raise ValueError("public experiment cells must be a list")
    for cell in cells:
        cell_value = _mapping(cell, "public cell")
        _exact_fields(cell_value, PUBLIC_CELL_FIELDS, "public cell")
    prediction_ids = [
        str(_mapping(cell, "public cell")["prediction_id"]) for cell in cells
    ]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("public experiment contains duplicate prediction IDs")
    matrix = _mapping(value.get("matrix"), "public matrix")
    expected = int(matrix.get("expected_predictions") or 0)
    expected_metrics = _metrics(cells, expected)
    expected_metrics["paired_bootstrap"] = (
        _confirmed_intervals(cells, str(value.get("id") or ""))
        if value.get("evidence_tier") == "confirmed"
        else None
    )
    if value.get("metrics") != expected_metrics:
        raise ValueError("public experiment metrics do not match canonical cells")
    if _canonical_json(value.get("groups")) != _canonical_json(_group_metrics(cells)):
        raise ValueError("public experiment groups do not match canonical cells")
    linked = sorted(
        {
            str(_mapping(cell, "public cell")["agent_link"])
            for cell in cells
            if cell["agent_link"]
        }
    )
    links = _mapping(value.get("links"), "public links")
    if links.get("evaluations") != linked:
        raise ValueError("public evaluation links do not match verified Agent cells")
    _validate_compatible_cohort(
        value, cells, complete=value.get("evidence_tier") in COMPLETE_TIERS
    )
    for url in _all_urls(value.get("links")) | _all_urls(value.get("provenance")):
        _validate_url(url)
    _reject_sensitive(value)
    digest = str(value.get("content_sha256") or "")
    if digest != _digest(
        {key: nested for key, nested in value.items() if key != "content_sha256"}
    ):
        raise ValueError("public experiment content digest does not match")


def _validate_public_experiment_v2(value: Mapping[str, Any]) -> None:
    allowed = {field.name for field in PublicExperimentV2.__dataclass_fields__.values()}
    _exact_fields(value, allowed, "public experiment V2")
    _validate_v2_contract(value, editorial=False)
    links = _mapping(value.get("links"), "public links")
    _exact_fields(links, LINK_FIELDS, "public links")
    for url in _all_urls(links) | _all_urls(value.get("provenance")):
        _validate_url(url)
    _reject_sensitive(value)
    digest = str(value.get("content_sha256") or "")
    if digest != _digest(
        {key: nested for key, nested in value.items() if key != "content_sha256"}
    ):
        raise ValueError("public experiment V2 content digest does not match")


def validate_experiment_index(value: Mapping[str, Any]) -> None:
    _exact_fields(
        value,
        {"schema_version", "experiments", "content_sha256"},
        "experiment index",
    )
    if value.get("schema_version") != EXPERIMENT_INDEX_SCHEMA_VERSION:
        raise ValueError("experiment index schema_version must be 1")
    experiments = value.get("experiments")
    if not isinstance(experiments, (list, tuple)):
        raise ValueError("experiment index experiments must be a list")
    expected_fields = {
        "id",
        "title",
        "summary",
        "evidence_tier",
        "decision_value",
        "status",
        "metrics",
        "models",
        "harnesses",
        "treatments",
        "content_sha256",
    }
    for experiment in experiments:
        _exact_fields(
            _mapping(experiment, "experiment index entry"),
            expected_fields,
            "experiment index entry",
        )
    digest = str(value.get("content_sha256") or "")
    body = {key: nested for key, nested in value.items() if key != "content_sha256"}
    if digest != _digest(body):
        raise ValueError("experiment index content digest does not match")


def _metrics(cells: Sequence[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    scored = [cell for cell in cells if cell["pass"] is not None]
    passed = sum(cell["pass"] is True for cell in scored)
    costs = [float(cell["cost_usd"]) for cell in cells if cell["cost_usd"] is not None]
    input_tokens = [
        cell["input_tokens"] for cell in cells if cell["input_tokens"] is not None
    ]
    output_tokens = [
        cell["output_tokens"] for cell in cells if cell["output_tokens"] is not None
    ]
    latencies = [
        float(cell["wall_time_sec"])
        for cell in cells
        if cell["wall_time_sec"] is not None
    ]
    tool_calls = [
        cell["tool_calls"] for cell in cells if cell["tool_calls"] is not None
    ]
    turns = [cell["turns"] for cell in cells if cell["turns"] is not None]
    recall = [
        float(cell["recall_at_10"])
        for cell in cells
        if cell["recall_at_10"] is not None
    ]
    reciprocal_ranks = [float(cell["mrr"]) for cell in cells if cell["mrr"] is not None]
    links = [cell for cell in cells if cell["agent_link"]]
    return {
        "predictions": len(cells),
        "expected_predictions": expected,
        "completion_rate": len(cells) / expected if expected else None,
        "scored_predictions": len(scored),
        "passed_predictions": passed,
        "pass_rate": passed / len(scored) if scored else None,
        "measured_cost_predictions": len(costs),
        "total_cost_usd": sum(costs) if len(costs) == len(cells) and cells else None,
        "mean_cost_usd": sum(costs) / len(costs)
        if len(costs) == len(cells) and cells
        else None,
        "measured_usage_predictions": sum(
            cell["input_tokens"] is not None and cell["output_tokens"] is not None
            for cell in cells
        ),
        "input_tokens": sum(input_tokens)
        if len(input_tokens) == len(cells) and cells
        else None,
        "output_tokens": sum(output_tokens)
        if len(output_tokens) == len(cells) and cells
        else None,
        "median_wall_time_sec": statistics.median(latencies) if latencies else None,
        "measured_latency_predictions": len(latencies),
        "tool_calls": sum(tool_calls)
        if len(tool_calls) == len(cells) and cells
        else None,
        "median_turns": statistics.median(turns) if turns else None,
        "mean_recall_at_10": statistics.fmean(recall) if recall else None,
        "mean_mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else None,
        "agent_links": len(links),
        "recoverable_errors": sum(int(cell["recoverable_errors"]) for cell in cells),
        "refusals": sum(int(cell["refusals"]) for cell in cells),
        "provider_errors": sum(int(cell["provider_errors"]) for cell in cells),
        "harness_errors": sum(int(cell["harness_errors"]) for cell in cells),
    }


def _group_metrics(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        key = (str(cell["model"]), str(cell["harness"]), str(cell["treatment"]))
        grouped.setdefault(key, []).append(cell)
    result: list[dict[str, Any]] = []
    for (model, harness, treatment), rows in sorted(grouped.items()):
        metrics = _metrics(rows, len(rows))
        result.append(
            {
                "model": model,
                "harness": harness,
                "treatment": treatment,
                "metrics": metrics,
                "tasks": [
                    {
                        "task_id": row["task_id"],
                        "trial_index": row["trial_index"],
                        "pass": row["pass"],
                        "reward": row["reward"],
                    }
                    for row in sorted(
                        rows,
                        key=lambda item: (
                            str(item["task_id"]),
                            int(item["trial_index"]),
                        ),
                    )
                ],
            }
        )
    return result


def _confirmed_intervals(
    cells: Sequence[Mapping[str, Any]], seed: str
) -> list[dict[str, Any]]:
    attempts = {int(cell["trial_index"]) for cell in cells}
    if len(attempts) < 2:
        raise ValueError("confirmed evidence requires replicated trials")
    treatments = {str(cell["treatment"]) for cell in cells}
    baseline_id = next(
        (value for value in ("none", "baseline") if value in treatments), None
    )
    if baseline_id is None:
        raise ValueError("confirmed evidence requires a baseline treatment")
    baseline = [
        {
            "comparison_example_id": cell["comparison_example_id"],
            "pass": cell["pass"],
        }
        for cell in cells
        if cell["treatment"] == baseline_id
    ]
    intervals: list[dict[str, Any]] = []
    for treatment in sorted(treatments - {baseline_id}):
        rows = [
            {
                "comparison_example_id": cell["comparison_example_id"],
                "pass": cell["pass"],
            }
            for cell in cells
            if cell["treatment"] == treatment
        ]
        low, high = _paired_delta_interval(
            rows,
            baseline,
            confidence=0.95,
            samples=2_000,
            seed=f"{seed}:{treatment}:{baseline_id}",
        )
        intervals.append(
            {
                "treatment": treatment,
                "baseline": baseline_id,
                "confidence": 0.95,
                "low": low,
                "high": high,
            }
        )
    return intervals


def _validate_compatible_cohort(
    editorial: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
) -> None:
    matrix = _mapping(editorial["matrix"], "matrix")
    allowed = {
        "models": set(_string_list(matrix.get("models"), "matrix.models")),
        "harnesses": set(_string_list(matrix.get("harnesses"), "matrix.harnesses")),
        "treatments": set(_string_list(matrix.get("treatments"), "matrix.treatments")),
        "tasks": set(_string_list(matrix.get("tasks"), "matrix.tasks")),
    }
    fields = {
        "models": "model",
        "harnesses": "harness",
        "treatments": "treatment",
        "tasks": "task_id",
    }
    for dimension, field in fields.items():
        observed = {str(cell[field]) for cell in cells}
        if observed - allowed[dimension]:
            raise ValueError(
                f"public rows contain incompatible {dimension}: {sorted(observed)}"
            )
    attempts = int(matrix.get("attempts") or 0)
    if attempts < 1:
        raise ValueError("matrix attempts must be positive")
    expected_trials = set(range(1, attempts + 1))
    observed_coordinates: set[tuple[str, str, str, str, int]] = set()
    cohorts = [_mapping(item, "matrix cohort") for item in matrix["cohorts"]]
    cohort_counts = {str(cohort["id"]): 0 for cohort in cohorts}
    for cell in cells:
        trial_index = int(cell["trial_index"])
        if trial_index not in expected_trials:
            raise ValueError("public row has a trial index outside the frozen matrix")
        coordinate = (
            str(cell["model"]),
            str(cell["harness"]),
            str(cell["treatment"]),
            str(cell["task_id"]),
            trial_index,
        )
        if coordinate in observed_coordinates:
            raise ValueError("public rows contain a duplicate frozen-matrix coordinate")
        observed_coordinates.add(coordinate)
        matches = [
            cohort
            for cohort in cohorts
            if (
                str(cell["model"]) in cohort["models"]
                and str(cell["harness"]) in cohort["harnesses"]
                and str(cell["treatment"]) in cohort["treatments"]
                and str(cell["task_id"]) in cohort["tasks"]
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "public row does not belong to a declared compatible cohort"
            )
        cohort_counts[str(matches[0]["id"])] += 1
    if complete:
        expected_counts = {
            str(cohort["id"]): int(cohort["expected_predictions"]) for cohort in cohorts
        }
        if cohort_counts != expected_counts:
            raise ValueError("public rows do not complete every declared cohort")


def _reject_sensitive(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if FORBIDDEN_KEY.search(key_text):
                raise ValueError(
                    f"public evidence contains forbidden field at {path}.{key_text}"
                )
            _reject_sensitive(nested, f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_sensitive(nested, f"{path}[{index}]")
    elif isinstance(value, str):
        if SECRET_VALUE.search(value):
            raise ValueError(f"public evidence contains a secret-like value at {path}")
        if value.startswith(
            ("/Users/", "/private/", "/home/", "~/", "file://")
        ) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ValueError(f"public evidence contains a local path at {path}")


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_URL_HOSTS:
        raise ValueError(f"public evidence URL is not approved: {url}")
    if parsed.username or parsed.password or parsed.query:
        raise ValueError(
            "public evidence URLs cannot contain credentials or query strings"
        )


def _all_urls(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for nested in value.values():
            found.update(_all_urls(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.update(_all_urls(nested))
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        found.add(value)
    return found


def _exact_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {sorted(unexpected)}")
    missing = allowed - set(value)
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings")
    return [str(item) for item in value]


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("public metric must be finite")
    return result


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("public text must be a non-empty string or null")
    return value


def _public_cost(row: Mapping[str, Any]) -> float | None:
    measured = [
        value
        for value in (
            _optional_number(row.get("cost_usd")),
            _optional_number(row.get("weave_total_cost_usd")),
        )
        if value is not None
    ]
    if any(value < 0 for value in measured):
        raise ValueError("public cost cannot be negative")
    return max(measured) if measured else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 0:
        raise ValueError("public count cannot be negative")
    return result


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("public boolean metric must be true, false, or null")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    encoded = _canonical_json(value).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(body, encoding="utf-8")


def import_reviewed_evidence(
    raw: Mapping[str, Any],
    *,
    outcome_key: str | None = None,
    publication_level: str = "full",
) -> dict[str, Any]:
    """Normalize an allowed public input without upgrading its evidence level."""
    if publication_level not in {"full", "summary", "inventory_only"}:
        raise ValueError("unsupported reviewed evidence publication level")
    if publication_level == "inventory_only":
        raise ValueError("inventory-only evidence cannot be promoted into a result")
    if raw.get("schema_version") == 1 and "rows" in raw:
        if publication_level != "full":
            raise ValueError("public-export rows require full publication")
        if not outcome_key:
            raise ValueError("public-export import requires a named outcome key")
        rows = raw.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("public-export rows must be non-empty")
        source_digest = str(raw.get("source_export_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", source_digest):
            raise ValueError("public-export source digest is invalid")
        cells = []
        for row in rows:
            value = _mapping(row, "public-export row")
            outcome = value.get(outcome_key)
            if outcome not in {0, 1, None}:
                raise ValueError("public-export outcome is not binary or unavailable")
            project = str(value.get("trace_project") or "")
            call_id = str(value.get("eval_predict_and_score_call_id") or "")
            evidence_link = (
                f"https://wandb.ai/{project}/weave/calls/{call_id}"
                if project and call_id
                else None
            )
            cells.append(
                {
                    "cell_id": str(value.get("prediction_id") or ""),
                    "run_id": str(value.get("run_id") or ""),
                    "candidate_id": str(value.get("candidate_id") or "")[:12],
                    "task_id": str(value.get("task_name") or ""),
                    "harness": str(value.get("harness") or ""),
                    "treatment": str(value.get("variant_id") or ""),
                    "attempt": int(value.get("trial_index") or 0),
                    "status": "terminal",
                    "outcome": None if outcome is None else bool(outcome),
                    "outcome_label": outcome_key,
                    "cost_usd": None,
                    "evidence_link": evidence_link,
                }
            )
        _reject_sensitive(cells)
        return {
            "kind": "public_export_v1",
            "source_digest": source_digest,
            "cells": cells,
        }
    if raw.get("schema_version") == 3 and raw.get("kind") in {
        "ComparisonResultV3",
        "ExperimentViewV3",
    }:
        result_digest = str(raw.get("result_digest") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", result_digest):
            raise ValueError("V3 comparison projection has no exact result digest")
        if publication_level != "summary":
            raise ValueError("V3 aggregate projection cannot be promoted to task rows")
        summary = _mapping(raw.get("summary"), "V3 comparison summary")
        _reject_sensitive(summary)
        return {
            "kind": str(raw["kind"]),
            "result_digest": result_digest,
            "summary": dict(summary),
            "cells": [],
        }
    raise ValueError("unsupported reviewed evidence envelope")


def load_inventory(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("inventory must be a mapping")
    _exact_fields(raw, {"schema_version", "entries"}, "inventory")
    if raw.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise ValueError("inventory schema_version must be 1")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("inventory entries must be non-empty")
    entry_ids: set[str] = set()
    run_ids: list[str] = []
    for raw_entry in entries:
        entry = _mapping(raw_entry, "inventory entry")
        _exact_fields(entry, INVENTORY_ENTRY_FIELDS, "inventory entry")
        entry_id = str(entry.get("id") or "")
        if not entry_id or entry_id in entry_ids:
            raise ValueError("inventory entry IDs must be non-empty and unique")
        entry_ids.add(entry_id)
        if entry.get("category") not in {
            "result",
            "contract",
            "historical",
            "cancelled_partial",
            "planned",
        }:
            raise ValueError("inventory category is unsupported")
        if entry.get("lifecycle_state") not in {
            "complete",
            "partial",
            "cancelled",
            "planned_unrun",
        }:
            raise ValueError("inventory lifecycle state is unsupported")
        if entry.get("publication_level") not in {
            "full",
            "summary",
            "inventory_only",
            "planned",
        }:
            raise ValueError("inventory publication level is unsupported")
        for field in ("label", "claim_boundary"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ValueError(f"inventory {field} must be non-empty text")
        entry_runs = _string_list(entry.get("run_ids"), "inventory run_ids")
        for run_id in entry_runs:
            if not re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{10}", run_id):
                raise ValueError("inventory run ID is malformed")
        run_ids.extend(entry_runs)
        _string_list(entry.get("study_ids"), "inventory study_ids")
        source = entry.get("source_reference")
        if source is not None:
            if not isinstance(source, str):
                raise ValueError("inventory source reference must be a URL")
            _validate_url(source)
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("inventory run IDs must appear exactly once")
    _reject_sensitive(raw)
    return raw


def inventory_publication(
    inventory: Mapping[str, Any], study_ids: set[str]
) -> dict[str, Any]:
    mapped = [
        study_id
        for entry in inventory["entries"]
        for study_id in entry["study_ids"]
    ]
    if set(mapped) != study_ids or len(mapped) != len(set(mapped)):
        raise ValueError("every detailed Atlas study must map to one inventory entry")
    body = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "entries": inventory["entries"],
    }
    return {**body, "content_sha256": _digest(body)}


def _write_or_validate_inventory(
    editorial_dir: Path,
    output: Path,
    study_ids: set[str],
    *,
    write: bool,
) -> None:
    path = editorial_dir.parent / "inventory.yaml"
    public_path = output / "inventory.json"
    if not path.exists() and not public_path.exists():
        return
    if not path.exists():
        raise ValueError("public inventory has no reviewed source")
    expected = inventory_publication(load_inventory(path), study_ids)
    if write:
        _write_json(public_path, expected)
        return
    actual = json.loads(public_path.read_text(encoding="utf-8"))
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError("public inventory does not match its reviewed source")


def _parse_paths(values: Sequence[str], label: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for value in values:
        experiment_id, separator, path = value.partition("=")
        if not separator or not experiment_id or not path:
            raise ValueError(f"{label} must use EXPERIMENT_ID=PATH")
        result.setdefault(experiment_id, []).append(Path(path))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editorial-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", action="append", default=[])
    parser.add_argument("--run-summary", action="append", default=[])
    parser.add_argument("--snapshot", action="append", default=[])
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    paths = list(args.editorial_dir.glob("*.yaml"))
    if not paths:
        parser.error("editorial directory has no YAML records")
    if args.validate_only:
        if args.rows or args.run_summary or args.snapshot:
            parser.error("--validate-only does not accept private evidence inputs")
        index = validate_publication(paths, args.output)
        print(
            json.dumps(
                {"experiments": len(index.experiments), "digest": index.content_sha256}
            )
        )
        return 0
    parsed_rows = _parse_paths(args.rows, "--rows")
    duplicate_rows = [key for key, values in parsed_rows.items() if len(values) != 1]
    if duplicate_rows:
        parser.error(f"experiments have multiple normalized exports: {duplicate_rows}")
    index = write_publication(
        paths,
        {key: values[0] for key, values in parsed_rows.items()},
        _parse_paths(args.run_summary, "--run-summary"),
        _parse_paths(args.snapshot, "--snapshot"),
        args.output,
        repo_root=args.repo_root.resolve(),
    )
    print(
        json.dumps(
            {"experiments": len(index.experiments), "digest": index.content_sha256}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
