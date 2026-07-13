from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from filelock import FileLock

from fugue.bench.scoring import latency_summary, score_evidence_paths
from fugue.model_plane import ModelRoute, resolve_model_route, trace_project_slug
from fugue.redaction import redact_value
from fugue.weave_support import initialize_weave


def export_rows(
    jobs: list[Path],
    *,
    fetch_weave: bool = False,
    weave_project: str | None = None,
) -> list[dict[str, Any]]:
    rows = [
        *[_row_from_trial(path) for job in jobs for path in _trial_result_paths(job)],
        *[row for job in jobs for row in _context_result_rows(job)],
        *[row for job in jobs for row in _cell_result_rows(job)],
    ]
    if fetch_weave:
        run_keys = list(
            dict.fromkeys(str(row["run_key"]) for row in rows if row.get("run_key"))
        )
        spans = fetch_weave_summaries(
            run_keys=run_keys,
            project=weave_project or _weave_project_from_env(),
        )
        for row in rows:
            if row.get("run_key"):
                row.update(spans.get(str(row["run_key"]), {}))
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    presets: list[str] | None = None,
    workloads: list[str] | None = None,
    systems: list[str] | None = None,
) -> list[dict[str, Any]]:
    filters = {
        "preset_id": set(presets or []),
        "workload_id": set(workloads or []),
        "context_system_id": set(systems or []),
    }
    return [
        row
        for row in rows
        if all(not values or str(row.get(key)) in values for key, values in filters.items())
    ]


def judge_qa_rows(
    rows: list[dict[str, Any]],
    *,
    model: str,
    env: dict[str, str],
    repo_root: Path,
) -> None:
    route = resolve_model_route(model, env)
    api_key = env.get(route.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{route.api_key_env} is required for QA judging")
    references = _qa_references(repo_root)
    with httpx.Client(timeout=120) as client:
        for row in rows:
            if row.get("record_type") != "trial" or row.get("workload_id") != "qa":
                continue
            task_id = str(row.get("task_name") or "").rsplit("/", 1)[-1]
            reference = references.get(task_id)
            answer = _trial_answer(row)
            if not reference or not answer:
                row["judge_error"] = "missing local reference or agent answer"
                continue
            started = time.perf_counter()
            try:
                payload, usage = _judge_request(
                    client,
                    route,
                    api_key,
                    reference=reference,
                    answer=answer,
                    evidence_paths=[str(item) for item in row.get("evidence_paths") or []],
                )
                row.update(
                    {
                        "judge_model": route.display_model,
                        "judge_correctness": _score(payload, "correctness"),
                        "judge_completeness": _score(payload, "completeness"),
                        "judge_groundedness": _score(payload, "groundedness"),
                        "judge_overall": _score(payload, "overall"),
                        "judge_reasoning": str(payload.get("reasoning") or "")[:4_000],
                        "judge_input_tokens": usage.get("input_tokens"),
                        "judge_output_tokens": usage.get("output_tokens"),
                        "judge_latency_ms": (time.perf_counter() - started) * 1000,
                        "judge_cost_usd": None,
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "judge_model": route.display_model,
                        "judge_latency_ms": (time.perf_counter() - started) * 1000,
                        "judge_error": f"{type(exc).__name__}: {exc}",
                    }
                )


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return True


def publish_to_weave(
    rows: list[dict[str, Any]],
    project: str | None = None,
    *,
    ledger_root: Path | None = None,
    republish: bool = False,
) -> int:
    project = project or _weave_project_from_env()
    weave = initialize_weave(project)
    logger_cls = getattr(weave, "EvaluationLogger", None)
    if logger_cls is None:
        raise RuntimeError("installed weave package has no EvaluationLogger")

    logger = logger_cls(model="fugue", dataset="fugue-context-evaluation")
    ledger = (ledger_root or Path(".fugue/runtime/publications")) / _safe_slug(project)
    ledger.mkdir(parents=True, exist_ok=True)
    published = 0
    for row in rows:
        logged = _weave_safe_row(row)
        publication_id = _publication_id(logged)
        marker = ledger / f"{publication_id}.json"
        with FileLock(marker.with_suffix(".lock"), timeout=120):
            if marker.is_file() and not republish:
                continue
            scores = {
                name: row[name]
                for name in (
                    "reward",
                    "mrr",
                    "ndcg_at_10",
                    "recall_at_1",
                    "recall_at_5",
                    "recall_at_10",
                    "recall_at_20",
                    "evidence_recall",
                    "citation_correctness",
                    "fact_recall",
                    "judge_correctness",
                    "judge_completeness",
                    "judge_groundedness",
                    "judge_overall",
                )
                if row.get(name) is not None
            }
            logger.log_example(
                inputs={
                    "publication_id": publication_id,
                    "task": row.get("task_name"),
                    "harness": row.get("harness"),
                    "context_system_id": row.get("context_system_id"),
                    "workload_id": row.get("workload_id"),
                    "variant_id": row.get("variant_id"),
                    "prompt_id": row.get("prompt_id"),
                },
                output=logged,
                scores=scores,
            )
            _write_publication_marker(marker, project, publication_id)
            published += 1
    return published


def _publication_id(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_publication_marker(path: Path, project: str, publication_id: str) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            {
                "project": project,
                "publication_id": publication_id,
                "published_at": datetime.now().isoformat(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temp, path)


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)


def _weave_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe = dict(row)
    if safe.get("query"):
        safe["query"] = str(safe["query"])[:1_000]
    if safe.get("exception_message"):
        safe["exception_message"] = str(safe["exception_message"])[:1_000]
    hits = []
    for value in (safe.get("hits") or [])[:20]:
        if not isinstance(value, dict):
            continue
        hits.append(
            {
                key: value.get(key)
                for key in ("path", "start_line", "end_line", "score")
                if value.get(key) is not None
            }
        )
    if "hits" in safe:
        safe["hits"] = hits
    safe.pop("trial_dir", None)
    safe.pop("judge_reasoning", None)
    return redact_value(safe)


def fetch_weave_summaries(
    *,
    run_keys: list[str],
    project: str,
    timeout_sec: float = 30.0,
) -> dict[str, dict[str, Any]]:
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is required to fetch Weave spans")
    base_url = os.environ.get("WEAVE_BASE_URL", "https://api.wandb.ai").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    summaries: dict[str, dict[str, Any]] = {}
    with httpx.Client(timeout=timeout_sec, headers=headers) as client:
        for run_key in run_keys:
            summaries[run_key] = _summarize_spans(
                _fetch_calls_spans(client, base_url, project, run_key)
                + _fetch_agents_spans(client, base_url, project, run_key)
            )
    return summaries


def _fetch_calls_spans(
    client: httpx.Client, base_url: str, project: str, run_key: str
) -> list[dict[str, Any]]:
    entity, name = project.split("/", 1)
    payload = {
        "project_id": f"{entity}/{name}",
        "filter": {
            "op_name": ".*",
            "trace_roots_only": False,
            "wb_user_ids": [],
        },
        "query": {
            "$expr": {
                "$eq": [
                    {"$getField": "attributes.fugue.run_key"},
                    {"$literal": run_key},
                ]
            }
        },
    }
    response = client.post(f"{base_url}/calls/stream_query", json=payload)
    if response.status_code >= 400:
        return []
    data = response.json()
    return data if isinstance(data, list) else data.get("calls", [])


def _fetch_agents_spans(
    client: httpx.Client, base_url: str, project: str, run_key: str
) -> list[dict[str, Any]]:
    payload = {"project_id": project, "filter": {"agent_name": run_key}}
    response = client.post(f"{base_url}/agents/spans/query", json=payload)
    if response.status_code >= 400:
        return []
    data = response.json()
    return data if isinstance(data, list) else data.get("spans", [])


def _summarize_spans(spans: list[dict[str, Any]]) -> dict[str, Any]:
    text = json.dumps(spans)
    return {
        "weave_span_count": len(spans),
        "context_read_count": text.count(".fugue-context")
        + text.count("AGENTS.md")
        + text.count("openwiki")
        + text.count("context_search"),
    }


def _trial_result_paths(job: Path) -> list[Path]:
    if job.is_file() and job.name == "result.json":
        return [job]
    if (job / "result.json").is_file() and (job / "agent").is_dir():
        return [job / "result.json"]
    return sorted(
        path
        for path in job.rglob("result.json")
        if path.parent != job and (path.parent / "agent").exists()
    )


def _row_from_trial(result_path: Path) -> dict[str, Any]:
    trial = json.loads(result_path.read_text())
    trial_dir = result_path.parent
    meta_path = trial_dir / "agent" / "fugue-meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    agent_result = trial.get("agent_result") or {}
    verifier_result = trial.get("verifier_result") or {}
    exception = trial.get("exception_info") or {}
    reward = (verifier_result.get("rewards") or {}).get("reward")
    started = _parse_time(trial.get("started_at"))
    finished = _parse_time(trial.get("finished_at"))
    wall_time = (finished - started).total_seconds() if started and finished else None
    context_events = _context_event_summary(trial_dir)
    evidence = _evidence_summary(
        trial_dir,
        trial.get("task_name"),
        meta.get("expected_evidence_paths") or {},
    )
    return {
        "schema_version": 1,
        "record_type": "trial",
        "run_key": meta.get("run_key") or trial.get("trial_name") or trial_dir.name,
        "run_id": meta.get("run_id"),
        "job_name": meta.get("job_name") or trial_dir.parent.name,
        "task_name": trial.get("task_name"),
        "trial_name": trial.get("trial_name") or trial_dir.name,
        "harness": meta.get("harness") or (trial.get("agent_info") or {}).get("name"),
        "experiment_id": meta.get("experiment_id"),
        "workload_id": meta.get("workload_id") or "harbor",
        "preset_id": meta.get("preset_id"),
        "run_name": meta.get("run_name"),
        "run_group": meta.get("run_group"),
        "variant_id": meta.get("variant_id"),
        "prompt_id": meta.get("prompt_id"),
        "context_system_id": meta.get("context_system_id", "none"),
        "context_version": meta.get("context_version"),
        "context_config_hash": meta.get("context_config_hash"),
        "context_cache_keys": meta.get("context_cache_keys", {}),
        "prompt_hashes": meta.get("prompt_hashes", {}),
        "skill_ids": meta.get("skill_ids", []),
        "skill_hashes": meta.get("skill_hashes", {}),
        "harbor_config": meta.get("harbor_config"),
        "harbor_environment": meta.get("harbor_environment"),
        "harbor_resources": meta.get("harbor_resources", {}),
        "agent_config_hash": meta.get("agent_config_hash"),
        "tags": meta.get("tags", []),
        "dataset": meta.get("dataset"),
        "manifest_path": meta.get("manifest_path"),
        "model_provider": meta.get("model_provider"),
        "builder_model": meta.get("builder_model"),
        "judge_model": meta.get("judge_model"),
        "model": meta.get("model")
        or ((trial.get("config") or {}).get("agent") or {}).get("model_name"),
        "trace_project": meta.get("trace_project")
        or (
            f"{meta.get('weave_entity')}/{meta.get('weave_project')}"
            if meta.get("weave_entity") and meta.get("weave_project")
            else None
        ),
        "reward": reward,
        "pass": reward == 1.0 if reward is not None else None,
        "wall_time_sec": wall_time,
        "n_input_tokens": agent_result.get("n_input_tokens"),
        "n_cache_tokens": agent_result.get("n_cache_tokens"),
        "n_output_tokens": agent_result.get("n_output_tokens"),
        "cost_usd": agent_result.get("cost_usd"),
        "exception_class": exception.get("exception_type"),
        "context_artifact": meta.get("context_artifact"),
        **context_events,
        **evidence,
        "session_ids": meta.get("session_ids", []),
        "trial_dir": trial_dir.as_posix(),
    }


def _context_result_rows(path: Path) -> list[dict[str, Any]]:
    candidates: list[Path]
    if path.is_file() and path.name == "context-results.jsonl":
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("context-results.jsonl"))
    else:
        candidates = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for line in candidate.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("trial_dir", candidate.parent.as_posix())
            rows.append(row)
    return rows


def _cell_result_rows(path: Path) -> list[dict[str, Any]]:
    candidates: list[Path]
    if path.is_file() and path.name == "cells.jsonl":
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("cells.jsonl"))
    else:
        candidates = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        latest: dict[str, dict[str, Any]] = {}
        for line in candidate.read_text(errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("cell_id"):
                latest[str(item["cell_id"])] = item
        for item in latest.values():
            rows.append(
                {
                    **item,
                    "record_type": "cell",
                    "task_name": item.get("task_id"),
                    "applicable": item.get("status") != "not_applicable",
                    "run_key": (
                        f"{item.get('run_id')}:{item.get('workload_id')}:cell:"
                        f"{item.get('task_id')}:{item.get('harness')}:"
                        f"{item.get('context_system_id')}"
                    ),
                    "trial_dir": candidate.parent.as_posix(),
                }
            )
    return rows


def _context_event_summary(trial_dir: Path) -> dict[str, Any]:
    paths = list(trial_dir.rglob("fugue-context-events.jsonl"))
    events: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    proxy_responses = [
        event for event in events if event.get("event") == "mcp_tool_response"
    ]
    provider_retrievals = [
        event for event in events if event.get("event") == "retrieve"
    ]
    logical_events = proxy_responses or provider_retrievals
    latencies = [
        float(
            (event.get("metrics") or {}).get("query_latency_ms")
            if (event.get("metrics") or {}).get("query_latency_ms") is not None
            else event.get("latency_ms")
        )
        for event in logical_events
        if (event.get("metrics") or {}).get("query_latency_ms") is not None
        or event.get("latency_ms") is not None
    ]
    latency_percentiles = latency_summary(latencies)
    first_context = [
        float(event["elapsed_ms"])
        for event in events
        if event.get("event") in {"retrieve", "mcp_tool_request"}
        and event.get("elapsed_ms") is not None
    ]
    return {
        "context_event_count": len(events),
        "context_call_count": len(logical_events),
        "context_query_count": len(logical_events),
        "context_proxy_event_count": sum(
            1 for event in events if event.get("layer") == "proxy"
        ),
        "context_upstream_event_count": sum(
            1 for event in events if event.get("layer") == "upstream"
        ),
        "context_provider_event_count": sum(
            1 for event in events if event.get("layer") == "provider"
        ),
        "context_error_count": sum(1 for event in events if event.get("error")),
        "context_query_latency_ms": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "context_query_latency_p50_ms": latency_percentiles["p50_ms"],
        "context_query_latency_p95_ms": latency_percentiles["p95_ms"],
        "time_to_first_context_ms": min(first_context) if first_context else None,
    }


def _evidence_summary(
    trial_dir: Path,
    task_name: str | None,
    expected_by_task: dict[str, Any],
) -> dict[str, Any]:
    observed: list[str] = []
    for path in trial_dir.rglob("fugue-evidence.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        values = (
            payload.get("paths", [])
            if isinstance(payload, dict)
            else payload
            if isinstance(payload, list)
            else []
        )
        for value in values[:100]:
            item = value.get("path") if isinstance(value, dict) else value
            if item:
                observed.append(str(item)[:1_000])
    expected: list[str] = []
    for key, values in expected_by_task.items():
        if task_name == key or str(task_name or "").endswith(f"/{key}"):
            expected = [str(value) for value in values]
            break
    scores = score_evidence_paths(expected, observed)
    return {
        "evidence_paths": list(dict.fromkeys(observed)),
        "expected_evidence_paths": expected,
        "evidence_recall": scores["evidence_recall"],
        "citation_correctness": scores["evidence_precision"],
    }


def _qa_references(repo_root: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    datasets_root = repo_root / ".fugue" / "cache" / "datasets"
    if not datasets_root.exists():
        return references
    for selection_path in datasets_root.rglob("selection.json"):
        source_path = selection_path.parent / "_source.jsonl"
        if not source_path.is_file():
            continue
        rows = [
            json.loads(line)
            for line in source_path.read_text().splitlines()
            if line.strip()
        ]
        for selected in json.loads(selection_path.read_text()):
            index = selected.get("source_index")
            if isinstance(index, int) and 0 <= index < len(rows):
                references[str(selected["task_id"])] = str(rows[index].get("answer") or "")
    return references


def _trial_answer(row: dict[str, Any]) -> str | None:
    trial_dir = Path(str(row.get("trial_dir") or ""))
    if not trial_dir.is_dir():
        return None
    candidates = list(trial_dir.rglob("fugue-answer.md"))
    if not candidates:
        return None
    value = candidates[0].read_text(errors="replace").strip()
    return value or None


def _judge_request(
    client: httpx.Client,
    route: ModelRoute,
    api_key: str,
    *,
    reference: str,
    answer: str,
    evidence_paths: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = f"""Evaluate a repository-grounded answer against the reference. Return only JSON with numeric fields correctness, completeness, groundedness, and overall from 0 to 1, plus a concise reasoning string. Groundedness should consider whether the cited repository paths plausibly support the answer. Do not require wording to match the reference.

REFERENCE:
{reference[:16_000]}

CANDIDATE:
{answer[:16_000]}

CITED PATHS:
{json.dumps(evidence_paths[:100])}
"""
    if route.messages_base_url:
        response = client.post(
            f"{route.messages_base_url}/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": route.model_id,
                "max_tokens": 800,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        body = response.json()
        text = "".join(
            str(item.get("text") or "")
            for item in body.get("content", [])
            if isinstance(item, dict)
        )
        raw_usage = body.get("usage") or {}
        usage = {
            "input_tokens": raw_usage.get("input_tokens"),
            "output_tokens": raw_usage.get("output_tokens"),
        }
    else:
        response = client.post(
            f"{route.chat_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": route.model_id,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        body = response.json()
        text = str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        raw_usage = body.get("usage") or {}
        usage = {
            "input_tokens": raw_usage.get("prompt_tokens"),
            "output_tokens": raw_usage.get("completion_tokens"),
        }
    return _json_object(text), usage


def _json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("judge returned no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    return value


def _score(payload: dict[str, Any], key: str) -> float:
    value = float(payload[key])
    if not 0 <= value <= 1:
        raise ValueError(f"judge {key} must be between 0 and 1")
    return value


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _weave_project_from_env() -> str:
    return trace_project_slug(os.environ)
