from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from fugue.model_plane import trace_project_slug


def export_rows(
    jobs: list[Path],
    *,
    fetch_weave: bool = False,
    weave_project: str | None = None,
) -> list[dict[str, Any]]:
    rows = [_row_from_trial(path) for job in jobs for path in _trial_result_paths(job)]
    if fetch_weave:
        spans = fetch_weave_summaries(
            run_keys=[row["run_key"] for row in rows],
            project=weave_project or _weave_project_from_env(),
        )
        for row in rows:
            row.update(spans.get(row["run_key"], {}))
    return rows


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


def publish_to_weave(rows: list[dict[str, Any]], project: str | None = None) -> None:
    try:
        import weave
    except ImportError as exc:
        raise RuntimeError("weave is not installed") from exc

    project = project or _weave_project_from_env()
    weave.init(project)
    logger_cls = getattr(weave, "EvaluationLogger", None)
    if logger_cls is None:
        raise RuntimeError("installed weave package has no EvaluationLogger")

    logger = logger_cls(model="fugue", dataset="repomembench")
    for row in rows:
        score = row.get("reward")
        logger.log_prediction(
            inputs={
                "task": row.get("task_name"),
                "harness": row.get("harness"),
                "feature_memory": row.get("feature_memory"),
                "variant_id": row.get("variant_id"),
                "prompt_id": row.get("prompt_id"),
            },
            output=row,
            scores={"reward": score} if score is not None else {},
            metadata=row,
        )


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
        "memory_read_count": text.count(".fugue-memory")
        + text.count("AGENTS.md")
        + text.count("openwiki"),
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
    return {
        "run_key": meta.get("run_key") or trial.get("trial_name") or trial_dir.name,
        "job_name": meta.get("job_name") or trial_dir.parent.name,
        "task_name": trial.get("task_name"),
        "trial_name": trial.get("trial_name") or trial_dir.name,
        "harness": meta.get("harness") or (trial.get("agent_info") or {}).get("name"),
        "experiment_id": meta.get("experiment_id"),
        "run_name": meta.get("run_name"),
        "run_group": meta.get("run_group"),
        "variant_id": meta.get("variant_id"),
        "prompt_id": meta.get("prompt_id"),
        "feature_memory": meta.get("feature_memory", "none"),
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
        "memory_artifact": meta.get("memory_artifact"),
        "session_ids": meta.get("session_ids", []),
        "trial_dir": trial_dir.as_posix(),
    }


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _weave_project_from_env() -> str:
    return trace_project_slug(os.environ)
