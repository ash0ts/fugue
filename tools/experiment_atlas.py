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
from typing import Any
from urllib.parse import urlparse

import yaml

from fugue.bench.reproducibility import verify_snapshot
from fugue.bench.scoring import _paired_delta_interval

PUBLIC_EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_INDEX_SCHEMA_VERSION = 1
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
    _exact_fields(raw, EDITORIAL_FIELDS, "editorial record")
    if raw.get("schema_version") != 1:
        raise ValueError("editorial schema_version must be 1")
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
    _reject_sensitive(raw)
    return raw


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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


def build_index(experiments: Iterable[PublicExperimentV1]) -> ExperimentIndexV1:
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
    experiments: list[PublicExperimentV1] = []
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
        elif snapshots:
            raise ValueError("planned or blocked experiments cannot attach run snapshots")
        public = build_public_experiment(editorial, rows, summaries)
        _write_json(output / "experiments" / f"{experiment_id}.json", public.to_dict())
        experiments.append(public)
    index = build_index(experiments)
    _write_json(output / "index.json", index.to_dict())
    return index


def _validated_provenance(
    editorial: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if not snapshots or any(not verify_snapshot(snapshot) for snapshot in snapshots):
        raise ValueError("complete public evidence requires valid immutable run snapshots")
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
    for snapshot in snapshots:
        run_id = str(snapshot["run_id"])
        snapshot_digests[run_id] = str(snapshot["snapshot_sha256"])
        request = _mapping(snapshot.get("request"), "snapshot request")
        manifests.add(str(request.get("manifest") or ""))
        experiment_ids.add(str(request.get("experiment_id") or ""))
        runtime = _mapping(snapshot.get("runtime"), "snapshot runtime")
        executions = _mapping(runtime.get("executions"), "snapshot executions")
        for execution in executions.values():
            source = _mapping(
                _mapping(execution, "snapshot execution").get("fugue_source"),
                "snapshot Fugue source",
            )
            if source.get("kind") != "git" or source.get("dirty") is not False:
                raise ValueError("public evidence requires a clean tracked Fugue source")
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
        raise ValueError("public runs do not share one dataset manifest")
    if experiment_ids != {str(_mapping(editorial["matrix"], "matrix")["experiment_id"])}:
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
        raise ValueError("normalized Agent rows do not cover the frozen matrix coordinates")

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
            raise ValueError(f"editorial provenance does not match run evidence: {field}")
    return derived


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
        "treatment": str(row.get("variant_id") or row.get("context_system_id") or "none"),
        "provider": str(row.get("model_provider") or row.get("provider") or ""),
        "model": str(row["model"]),
        "status": str(row.get("status") or "unknown"),
        "pass": _optional_bool(row.get("pass")),
        "reward": _optional_number(row.get("reward")),
        "wall_time_sec": _optional_number(row.get("wall_time_sec")),
        "cost_usd": _optional_number(row.get("cost_usd")),
        "input_tokens": _optional_int(row.get("n_input_tokens")),
        "output_tokens": _optional_int(row.get("n_output_tokens")),
        "tool_calls": _optional_int(row.get("weave_tool_call_count")),
        "turns": _optional_int(row.get("weave_turn_count")),
        "recoverable_errors": int(row.get("recoverable_error_count") or 0),
        "refusals": int(row.get("refusal_count") or 0),
        "provider_errors": int(row.get("provider_error_count") or 0),
        "harness_errors": int(
            row.get("harness_error_count") or row.get("harness_adapter_error_count") or 0
        ),
        "context_registered": _optional_bool(row.get("context_registered")),
        "context_invoked": _optional_bool(row.get("context_invoked")),
        "context_invocation_count": _optional_int(row.get("context_invocation_count")),
        "recall_at_10": _optional_number(row.get("recall_at_10")),
        "mrr": _optional_number(row.get("mrr")),
        "agent_link": link,
    }
    assert set(cell) == PUBLIC_CELL_FIELDS
    _reject_sensitive(cell)
    return cell


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
    if not observed_runs or observed_runs - expected_runs or observed_runs - summary_runs:
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
    prediction_ids = [str(_mapping(cell, "public cell")["prediction_id"]) for cell in cells]
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
        {str(_mapping(cell, "public cell")["agent_link"]) for cell in cells if cell["agent_link"]}
    )
    links = _mapping(value.get("links"), "public links")
    if links.get("evaluations") != linked:
        raise ValueError("public evaluation links do not match verified Agent cells")
    _validate_compatible_cohort(value, cells, complete=value.get("evidence_tier") in COMPLETE_TIERS)
    for url in _all_urls(value.get("links")) | _all_urls(value.get("provenance")):
        _validate_url(url)
    _reject_sensitive(value)
    digest = str(value.get("content_sha256") or "")
    if digest != _digest({key: nested for key, nested in value.items() if key != "content_sha256"}):
        raise ValueError("public experiment content digest does not match")


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
        _exact_fields(_mapping(experiment, "experiment index entry"), expected_fields, "experiment index entry")
    digest = str(value.get("content_sha256") or "")
    body = {key: nested for key, nested in value.items() if key != "content_sha256"}
    if digest != _digest(body):
        raise ValueError("experiment index content digest does not match")


def _metrics(cells: Sequence[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    scored = [cell for cell in cells if cell["pass"] is not None]
    passed = sum(cell["pass"] is True for cell in scored)
    costs = [float(cell["cost_usd"]) for cell in cells if cell["cost_usd"] is not None]
    input_tokens = [cell["input_tokens"] for cell in cells if cell["input_tokens"] is not None]
    output_tokens = [cell["output_tokens"] for cell in cells if cell["output_tokens"] is not None]
    latencies = [float(cell["wall_time_sec"]) for cell in cells if cell["wall_time_sec"] is not None]
    tool_calls = [cell["tool_calls"] for cell in cells if cell["tool_calls"] is not None]
    turns = [cell["turns"] for cell in cells if cell["turns"] is not None]
    recall = [float(cell["recall_at_10"]) for cell in cells if cell["recall_at_10"] is not None]
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
        "mean_cost_usd": sum(costs) / len(costs) if len(costs) == len(cells) and cells else None,
        "measured_usage_predictions": sum(
            cell["input_tokens"] is not None and cell["output_tokens"] is not None
            for cell in cells
        ),
        "input_tokens": sum(input_tokens) if len(input_tokens) == len(cells) and cells else None,
        "output_tokens": sum(output_tokens) if len(output_tokens) == len(cells) and cells else None,
        "median_wall_time_sec": statistics.median(latencies) if latencies else None,
        "measured_latency_predictions": len(latencies),
        "tool_calls": sum(tool_calls) if len(tool_calls) == len(cells) and cells else None,
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
                        rows, key=lambda item: (str(item["task_id"]), int(item["trial_index"]))
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
            raise ValueError(f"public rows contain incompatible {dimension}: {sorted(observed)}")
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
            raise ValueError("public row does not belong to a declared compatible cohort")
        cohort_counts[str(matches[0]["id"])] += 1
    if complete:
        expected_counts = {
            str(cohort["id"]): int(cohort["expected_predictions"])
            for cohort in cohorts
        }
        if cohort_counts != expected_counts:
            raise ValueError("public rows do not complete every declared cohort")


def _reject_sensitive(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if FORBIDDEN_KEY.search(key_text):
                raise ValueError(f"public evidence contains forbidden field at {path}.{key_text}")
            _reject_sensitive(nested, f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_sensitive(nested, f"{path}[{index}]")
    elif isinstance(value, str):
        if SECRET_VALUE.search(value):
            raise ValueError(f"public evidence contains a secret-like value at {path}")
        if value.startswith(("/Users/", "/private/", "/home/", "~/", "file://")) or re.match(
            r"^[A-Za-z]:[\\/]", value
        ):
            raise ValueError(f"public evidence contains a local path at {path}")


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_URL_HOSTS:
        raise ValueError(f"public evidence URL is not approved: {url}")
    if parsed.username or parsed.password or parsed.query:
        raise ValueError("public evidence URLs cannot contain credentials or query strings")


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
    args = parser.parse_args(argv)
    paths = list(args.editorial_dir.glob("*.yaml"))
    if not paths:
        parser.error("editorial directory has no YAML records")
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
    print(json.dumps({"experiments": len(index.experiments), "digest": index.content_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
