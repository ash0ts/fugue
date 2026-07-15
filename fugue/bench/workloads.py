from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from fugue.bench.context import (
    ContextEvent,
    ContextRuntime,
    ContextSystemSpec,
    PreparedContext,
    RetrievalQuery,
    checkout_repository,
    get_context_system,
    load_provider,
    prepare_context,
    query_context,
)
from fugue.bench.library import validate_id
from fugue.bench.scoring import score_fact_recall, score_retrieval
from fugue.model_plane import resolve_model_route, trace_project_slug
from fugue.weave_support import trace_async_operation


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    repo: str
    commit: str
    query: str
    expected_paths: tuple[str, ...]
    family: str | None = None


@dataclass(frozen=True)
class SequenceProbe:
    id: str
    after_episode: int
    query: str
    expected_paths: tuple[str, ...] = ()
    expected_facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class SequenceCase:
    id: str
    repo: str
    commit: str
    events: tuple[ContextEvent, ...]
    probes: tuple[SequenceProbe, ...]


@dataclass(frozen=True)
class WorkloadDataset:
    id: str
    runner: str
    retrieval_cases: tuple[RetrievalCase, ...] = ()
    sequence_cases: tuple[SequenceCase, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)


def load_workload_dataset(path: Path) -> WorkloadDataset:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: workload dataset must be a mapping")
    runner = str(raw.get("runner") or "retrieval")
    if runner not in {"retrieval", "sequence"}:
        raise ValueError(f"{path}: unsupported workload runner {runner}")
    retrieval_cases = tuple(_retrieval_case(item) for item in raw.get("cases", []))
    sequence_cases = tuple(_sequence_case(item) for item in raw.get("sequences", []))
    _require_unique_ids([item.id for item in retrieval_cases], "retrieval case", path)
    _require_unique_ids([item.id for item in sequence_cases], "sequence", path)
    if runner == "retrieval" and not retrieval_cases:
        raise ValueError(f"{path}: retrieval workload needs cases")
    if runner == "sequence" and not sequence_cases:
        raise ValueError(f"{path}: sequence workload needs sequences")
    return WorkloadDataset(
        id=validate_id(raw.get("id") or path.stem, kind="workload dataset id"),
        runner=runner,
        retrieval_cases=retrieval_cases,
        sequence_cases=sequence_cases,
        source=dict(raw.get("source") or {}),
    )


async def run_retrieval_workload(
    *,
    dataset: WorkloadDataset,
    system_id: str,
    runtime: ContextRuntime,
    experiment_id: str,
    preset_id: str,
    run_id: str,
    attempts: int = 1,
    limit: int | None = None,
    rebuild: bool = False,
) -> list[dict[str, Any]]:
    _validate_workload_counts(attempts=attempts, limit=limit)
    spec = get_context_system(system_id, runtime.repo_root)
    available_cases = await asyncio.to_thread(
        _materialized_retrieval_cases, dataset, runtime, preset_id
    )
    cases = available_cases[:limit] if limit else available_cases
    rows: list[dict[str, Any]] = []
    for case in cases:
        if "retrieve" not in spec.capabilities:
            rows.append(
                _base_row(
                    record_type="retrieval",
                    experiment_id=experiment_id,
                    preset_id=preset_id,
                    workload_id=dataset.id,
                    system_id=system_id,
                    task_id=case.id,
                    attempt=1,
                    applicable=False,
                    skip_reason="context system has no ranked retrieval capability",
                )
            )
            continue
        snapshot = await asyncio.to_thread(
            checkout_repository,
            task_id=case.id,
            repo=case.repo,
            commit=case.commit,
            checkout_root=runtime.cache_root / "checkouts",
            dataset_id=dataset.id,
        )
        trace = _trace_fields(
            experiment_id,
            preset_id,
            dataset.id,
            system_id,
            case.id,
            0,
        )
        prepared = await trace_async_operation(
            "fugue.context.prepare",
            trace,
            runtime.env,
            lambda spec=spec, snapshot=snapshot: prepare_context(
                spec, snapshot, runtime, rebuild=rebuild
            ),
            lambda value: {
                "cache_key": value.cache_key,
                "cache_hit": value.cache_hit,
                **value.metrics,
            },
        )
        rows.append(
            {
                **_base_row(
                    record_type="preparation",
                    experiment_id=experiment_id,
                    preset_id=preset_id,
                    workload_id=dataset.id,
                    system_id=system_id,
                    task_id=case.id,
                    attempt=0,
                ),
                "context_version": spec.version,
                "context_cache_key": prepared.cache_key,
                "cache_hit": prepared.cache_hit,
                **prepared.metrics,
                "harness": "direct",
            }
        )
        for attempt in range(1, attempts + 1):
            query = RetrievalQuery(
                id=case.id,
                text=case.query,
                expected_paths=case.expected_paths,
            )
            try:
                hits, telemetry = await trace_async_operation(
                    "fugue.context.retrieve",
                    {**trace, "attempt": attempt, "query_id": query.id},
                    runtime.env,
                    lambda spec=spec, query=query, prepared=prepared: query_context(
                        spec, query, prepared, runtime
                    ),
                    lambda value: {
                        **value[1],
                        "paths": [hit.path for hit in value[0][:20]],
                    },
                )
                score = await trace_async_operation(
                    "fugue.context.score_retrieval",
                    {**trace, "attempt": attempt, "query_id": query.id},
                    runtime.env,
                    lambda query=query, hits=hits: _async_value(
                        score_retrieval(query, hits)
                    ),
                    lambda value: value,
                )
                row = {
                    **_base_row(
                        record_type="retrieval",
                        experiment_id=experiment_id,
                        preset_id=preset_id,
                        workload_id=dataset.id,
                        system_id=system_id,
                        task_id=case.id,
                        attempt=attempt,
                    ),
                    "context_version": spec.version,
                    "context_cache_key": prepared.cache_key,
                    "query_id": query.id,
                    "query_family": case.family,
                    "query": query.text,
                    "expected_paths": list(query.expected_paths),
                    "hits": [asdict(hit) for hit in hits],
                    **telemetry,
                    **score,
                    "harness": "direct",
                }
            except Exception as exc:
                row = {
                    **_base_row(
                        record_type="retrieval",
                        experiment_id=experiment_id,
                        preset_id=preset_id,
                        workload_id=dataset.id,
                        system_id=system_id,
                        task_id=case.id,
                        attempt=attempt,
                    ),
                    "exception_class": type(exc).__name__,
                    "exception_message": str(exc),
                    "harness": "direct",
                }
            rows.append(row)
    _add_runtime_correlation(rows, spec, runtime, run_id)
    await asyncio.to_thread(_write_rows, runtime.repo_root, run_id, rows)
    return rows


async def run_sequence_workload(
    *,
    dataset: WorkloadDataset,
    system_id: str,
    runtime: ContextRuntime,
    experiment_id: str,
    preset_id: str,
    run_id: str,
    attempts: int = 1,
    limit: int | None = None,
    rebuild: bool = False,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    _validate_workload_counts(attempts=attempts, limit=limit)
    if concurrency < 1:
        raise ValueError("sequence concurrency must be positive")
    spec = get_context_system(system_id, runtime.repo_root)
    cases = dataset.sequence_cases[:limit] if limit else dataset.sequence_cases
    rows: list[dict[str, Any]] = []
    if "ingest" not in spec.capabilities:
        rows = [
            _base_row(
                record_type="episode",
                experiment_id=experiment_id,
                preset_id=preset_id,
                workload_id=dataset.id,
                system_id=system_id,
                task_id=case.id,
                attempt=1,
                applicable=False,
                skip_reason="context system has no longitudinal ingestion capability",
            )
            for case in cases
        ]
        _add_runtime_correlation(rows, spec, runtime, run_id)
        await asyncio.to_thread(_write_rows, runtime.repo_root, run_id, rows)
        return rows

    semaphore = asyncio.Semaphore(concurrency)

    async def prepare_case(
        case: SequenceCase,
    ) -> tuple[dict[str, Any], PreparedContext, list[list[dict[str, Any]]]]:
        snapshot = await asyncio.to_thread(
            checkout_repository,
            task_id=case.id,
            repo=case.repo,
            commit=case.commit,
            checkout_root=runtime.cache_root / "checkouts",
            dataset_id=dataset.id,
        )
        trace = _trace_fields(
            experiment_id,
            preset_id,
            dataset.id,
            system_id,
            case.id,
            0,
        )
        prepared = await trace_async_operation(
            "fugue.context.prepare",
            trace,
            runtime.env,
            lambda spec=spec, snapshot=snapshot: prepare_context(
                spec, snapshot, runtime, rebuild=rebuild
            ),
            lambda value: {
                "cache_key": value.cache_key,
                "cache_hit": value.cache_hit,
                **value.metrics,
            },
        )
        preparation = {
            **_base_row(
                record_type="preparation",
                experiment_id=experiment_id,
                preset_id=preset_id,
                workload_id=dataset.id,
                system_id=system_id,
                task_id=case.id,
                attempt=0,
            ),
            "context_version": spec.version,
            "context_cache_key": prepared.cache_key,
            "cache_hit": prepared.cache_hit,
            **prepared.metrics,
            "harness": "sequence",
            "execution_kind": "provider_diagnostic",
        }

        async def run_attempt(attempt: int) -> list[dict[str, Any]]:
            async with semaphore:
                return await _run_sequence_cohort(
                    case=case,
                    attempt=attempt,
                    spec=spec,
                    prepared=prepared,
                    runtime=runtime,
                    trace=trace,
                    run_id=run_id,
                    experiment_id=experiment_id,
                    preset_id=preset_id,
                    workload_id=dataset.id,
                    system_id=system_id,
                )

        cohorts = await asyncio.gather(
            *(run_attempt(attempt) for attempt in range(1, attempts + 1))
        )
        return preparation, prepared, list(cohorts)

    prepared_cases = await asyncio.gather(*(prepare_case(case) for case in cases))
    for preparation, _, cohorts in prepared_cases:
        rows.append(preparation)
        for cohort_rows in cohorts:
            rows.extend(cohort_rows)
    _add_runtime_correlation(rows, spec, runtime, run_id)
    await asyncio.to_thread(_write_rows, runtime.repo_root, run_id, rows)
    return rows


async def _run_sequence_cohort(
    *,
    case: SequenceCase,
    attempt: int,
    spec: ContextSystemSpec,
    prepared: PreparedContext,
    runtime: ContextRuntime,
    trace: dict[str, Any],
    run_id: str,
    experiment_id: str,
    preset_id: str,
    workload_id: str,
    system_id: str,
) -> list[dict[str, Any]]:
    namespace = (
        runtime.repo_root
        / ".fugue"
        / "runtime"
        / run_id
        / "sequences"
        / system_id
        / case.id
        / str(attempt)
    )
    provider = load_provider(spec)
    rows: list[dict[str, Any]] = []
    previous_storage_bytes = 0
    try:
        for event in sorted(case.events, key=lambda item: item.episode):
            started = time.perf_counter()
            metrics = await trace_async_operation(
                "fugue.context.ingest",
                {
                    **trace,
                    "attempt": attempt,
                    "episode": event.episode,
                    "event_kind": event.kind,
                },
                runtime.env,
                lambda provider=provider, event=event, namespace=namespace: provider.ingest(
                    spec, event, namespace, runtime
                ),
                lambda value: value,
            )
            rows.append(
                {
                    **_base_row(
                        record_type="episode",
                        experiment_id=experiment_id,
                        preset_id=preset_id,
                        workload_id=workload_id,
                        system_id=system_id,
                        task_id=case.id,
                        attempt=attempt,
                    ),
                    "sequence_id": case.id,
                    "episode": event.episode,
                    "event_kind": event.kind,
                    "write_latency_ms": (time.perf_counter() - started) * 1000,
                    "storage_growth_bytes": max(
                        0,
                        int(metrics.get("storage_bytes") or 0) - previous_storage_bytes,
                    ),
                    "harness": "sequence",
                    "execution_kind": "provider_diagnostic",
                    **metrics,
                }
            )
            previous_storage_bytes = int(
                metrics.get("storage_bytes") or previous_storage_bytes
            )
            for probe in (
                probe for probe in case.probes if probe.after_episode == event.episode
            ):
                if "retrieve" not in spec.capabilities:
                    continue
                query = RetrievalQuery(
                    id=probe.id,
                    text=probe.query,
                    expected_paths=probe.expected_paths,
                )
                started = time.perf_counter()
                hits = await trace_async_operation(
                    "fugue.context.retrieve",
                    {
                        **trace,
                        "attempt": attempt,
                        "episode": event.episode,
                        "query_id": query.id,
                    },
                    runtime.env,
                    lambda provider=provider, query=query, prepared=prepared, namespace=namespace: provider.retrieve(
                        spec,
                        query,
                        replace(prepared, path=namespace),
                        runtime,
                    ),
                    lambda value: {
                        "result_count": len(value),
                        "paths": [hit.path for hit in value[:20]],
                    },
                )
                rows.append(
                    {
                        **_base_row(
                            record_type="retrieval",
                            experiment_id=experiment_id,
                            preset_id=preset_id,
                            workload_id=workload_id,
                            system_id=system_id,
                            task_id=case.id,
                            attempt=attempt,
                        ),
                        "sequence_id": case.id,
                        "episode": event.episode,
                        "query_id": probe.id,
                        "hits": [asdict(hit) for hit in hits],
                        "query_latency_ms": (time.perf_counter() - started) * 1000,
                        "result_count": len(hits),
                        **score_retrieval(query, hits),
                        **score_fact_recall(
                            probe.expected_facts,
                            [hit.text or "" for hit in hits],
                        ),
                        "harness": "sequence",
                        "execution_kind": "provider_diagnostic",
                    }
                )
    finally:
        await provider.close()
    return rows


def _retrieval_case(raw: Any) -> RetrievalCase:
    if not isinstance(raw, dict):
        raise ValueError("retrieval case must be a mapping")
    return RetrievalCase(
        id=validate_id(raw["id"], kind="retrieval case id"),
        repo=str(raw["repo"]),
        commit=str(raw["commit"]),
        query=str(raw["query"]),
        expected_paths=tuple(str(item) for item in raw.get("expected_paths", [])),
        family=str(raw["family"]) if raw.get("family") else None,
    )


def _sequence_case(raw: Any) -> SequenceCase:
    if not isinstance(raw, dict):
        raise ValueError("sequence must be a mapping")
    sequence_id = validate_id(raw["id"], kind="sequence id")
    events = tuple(
        ContextEvent(
            sequence_id=sequence_id,
            episode=int(item["episode"]),
            kind=str(item.get("kind") or "observation"),
            content=str(item["content"]),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in raw.get("events", [])
    )
    probes = tuple(
        SequenceProbe(
            id=validate_id(item["id"], kind="sequence probe id"),
            after_episode=int(item["after_episode"]),
            query=str(item["query"]),
            expected_paths=tuple(str(value) for value in item.get("expected_paths", [])),
            expected_facts=tuple(str(value) for value in item.get("expected_facts", [])),
        )
        for item in raw.get("probes", [])
    )
    _require_unique_ids(
        [str(event.episode) for event in events],
        f"sequence {sequence_id} episode",
    )
    _require_unique_ids(
        [probe.id for probe in probes], f"sequence {sequence_id} probe"
    )
    if any(event.episode < 1 for event in events):
        raise ValueError(f"sequence {sequence_id} episodes must be positive")
    episode_ids = {event.episode for event in events}
    if any(probe.after_episode not in episode_ids for probe in probes):
        raise ValueError(
            f"sequence {sequence_id} probes must reference an existing episode"
        )
    return SequenceCase(
        id=sequence_id,
        repo=str(raw["repo"]),
        commit=str(raw["commit"]),
        events=events,
        probes=probes,
    )


def _validate_workload_counts(*, attempts: int, limit: int | None) -> None:
    if attempts < 1:
        raise ValueError("workload attempts must be positive")
    if limit is not None and limit < 1:
        raise ValueError("workload limit must be positive")


def _require_unique_ids(
    values: list[str], kind: str, path: Path | None = None
) -> None:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        prefix = f"{path}: " if path else ""
        raise ValueError(f"{prefix}duplicate {kind} id(s): {', '.join(duplicates)}")


def _base_row(
    *,
    record_type: str,
    experiment_id: str,
    preset_id: str,
    workload_id: str,
    system_id: str,
    task_id: str,
    attempt: int,
    applicable: bool = True,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": record_type,
        "experiment_id": experiment_id,
        "preset_id": preset_id,
        "workload_id": workload_id,
        "context_system_id": system_id,
        "task_name": task_id,
        "attempt": attempt,
        "applicable": applicable,
        "skip_reason": skip_reason,
    }


def _write_rows(repo_root: Path, run_id: str, rows: list[dict[str, Any]]) -> Path:
    path = repo_root / ".fugue" / "runtime" / run_id / "context-results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return path


def _add_runtime_correlation(
    rows: list[dict[str, Any]],
    spec: ContextSystemSpec,
    runtime: ContextRuntime,
    run_id: str,
) -> None:
    model = runtime.env.get("FUGUE_BUILDER_MODEL") or runtime.env.get("FUGUE_MODEL")
    route = resolve_model_route(model, runtime.env) if model else None
    config_hash = hashlib.sha256(
        json.dumps(spec.config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    trace_project = trace_project_slug(runtime.env)
    for row in rows:
        row.update(
            {
                "run_id": run_id,
                "run_key": (
                    f"{run_id}:{row.get('workload_id')}:{spec.id}:"
                    f"{row.get('task_name')}:{row.get('attempt')}:"
                    f"{row.get('record_type')}:"
                    f"{row.get('episode') or 0}:{row.get('query_id') or '-'}"
                ),
                "variant_id": spec.id,
                "context_version": spec.version,
                "context_config_hash": config_hash,
                "context_transport": runtime.env.get(
                    "FUGUE_CONTEXT_TRANSPORT", "portable"
                ),
                "trace_project": trace_project,
                "model_role": "context_builder",
                "builder_model": route.display_model if route else None,
                "model_provider": route.provider if route else None,
                "embedding_model": spec.config.get("embedding_model"),
            }
        )


def _trace_fields(
    experiment_id: str,
    preset_id: str,
    workload_id: str,
    system_id: str,
    task_id: str,
    attempt: int,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "preset_id": preset_id,
        "workload_id": workload_id,
        "context_system_id": system_id,
        "task_id": task_id,
        "attempt": attempt,
    }


async def _async_value(value: Any) -> Any:
    return value


def _materialized_retrieval_cases(
    dataset: WorkloadDataset, runtime: ContextRuntime, preset_id: str
) -> tuple[RetrievalCase, ...]:
    if preset_id != "full" or not dataset.source.get("materialize_command"):
        return dataset.retrieval_cases
    version = str(dataset.source.get("version") or "v1")
    root = runtime.cache_root / "datasets" / dataset.id / version
    candidates = [
        root / "benchmark" / version / "samples.jsonl",
        root / "data" / "benchmark" / version / "samples.jsonl",
    ]
    samples_path = next((path for path in candidates if path.is_file()), None)
    if samples_path is None:
        root.mkdir(parents=True, exist_ok=True)
        command = [
            str(token).format(output=root.as_posix(), version=version)
            for token in dataset.source["materialize_command"]
        ]
        subprocess.run(command, check=True, env=runtime.env)
        samples_path = next((path for path in candidates if path.is_file()), None)
    if samples_path is None:
        raise FileNotFoundError(
            f"{dataset.id} materializer did not create benchmark/{version}/samples.jsonl"
        )
    cases = tuple(
        _arb_case(json.loads(line), index)
        for index, line in enumerate(samples_path.read_text().splitlines(), start=1)
        if line.strip()
    )
    expected_count = int((dataset.source.get("counts") or {}).get("full") or 0)
    if expected_count and len(cases) != expected_count:
        raise ValueError(
            f"{dataset.id} expected {expected_count} samples, found {len(cases)}"
        )
    return cases


def _arb_case(raw: dict[str, Any], index: int) -> RetrievalCase:
    repo = _first(raw, "repo", "repository", "repo_name", "repo_slug")
    commit = _first(raw, "base_commit", "commit", "repo_commit")
    query = _first(raw, "query", "query_text", "prompt", "description")
    expected = (
        raw.get("expected_paths")
        or raw.get("gold_files")
        or raw.get("gold_paths")
        or raw.get("target_files")
        or []
    )
    if isinstance(expected, str):
        expected = [expected]
    if not repo or not commit or not query or not expected:
        raise ValueError(
            f"Agent Retrieval Bench sample {index} is missing repo, commit, query, or gold files"
        )
    return RetrievalCase(
        id=str(raw.get("id") or raw.get("sample_id") or f"arb-{index:03d}"),
        repo=str(repo),
        commit=str(commit),
        query=str(query),
        expected_paths=tuple(str(item) for item in expected),
        family=str(raw.get("family") or raw.get("task_type") or "unknown"),
    )


def _first(raw: dict[str, Any], *keys: str) -> Any:
    return next((raw[key] for key in keys if raw.get(key) not in (None, "")), None)
