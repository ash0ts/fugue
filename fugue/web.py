from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from fastapi import Request
except ImportError:  # pragma: no cover - web extra not installed
    Request = Any  # type: ignore

from fugue.bench.cli import _load_env
from fugue.bench.export import export_rows
from fugue.bench.job_config import RenderedJob, preview_jobs, render_jobs
from fugue.bench.library import (
    ExperimentSpec,
    experiment_from_data,
    experiment_to_yaml,
    experiment_with_overrides,
    get_experiment,
    get_experiment_text,
    get_prompt,
    get_skill,
    list_experiments,
    list_prompts,
    list_skills,
    save_experiment,
    save_experiment_data,
    save_prompt,
    save_skill,
)
from fugue.bench.manifest import load_manifest
from fugue.bridge import bridge_status
from fugue.model_plane import (
    DEFAULT_MODEL,
    env_presence,
    resolve_model_route,
    select_model,
    trace_project_slug,
)
from fugue.web_jobs import job_detail, list_jobs, start_job, tail_job_events

STATIC_DIR = Path(__file__).resolve().parent / "web_static"


def run_web(host: str = "127.0.0.1", port: int = 8765) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('install web dependencies with: uv pip install -e ".[web]"') from exc

    uvicorn.run("fugue.web:create_app", factory=True, host=host, port=port)


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError('install web dependencies with: uv pip install -e ".[web]"') from exc

    app = FastAPI(title="Fugue")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        return _status_payload()

    @app.get("/api/summary")
    def api_summary() -> dict[str, Any]:
        status = _status_payload()
        manifest = _manifest_payload("datasets/pilot.yaml")
        rows = _safe_export_rows(Path("jobs"))
        jobs = list_jobs()
        result_summary = _summarize_rows(rows)
        readiness = {
            "trace": bool(status.get("weave_project"))
            and status["keys"].get("WANDB_API_KEY", False),
            "model": _model_key_ready(status),
            "bridge": bool((status.get("bridge") or {}).get("ok")),
            "manifest": manifest["counts"]["tasks"] > 0
            and manifest["counts"]["harnesses"] > 0,
        }
        return {
            "status": status,
            "manifest": manifest,
            "readiness": readiness,
            "jobs": {"latest": jobs[0] if jobs else None, "count": len(jobs)},
            "results": result_summary,
        }

    @app.get("/api/manifest")
    def api_manifest(path: str = "datasets/pilot.yaml") -> dict[str, Any]:
        return _manifest_payload(path)

    @app.get("/api/library")
    def api_library() -> dict[str, Any]:
        return _library_payload()

    @app.get("/api/prompts/{item_id}")
    def api_prompt(item_id: str) -> dict[str, Any]:
        try:
            return _dataclass_payload(get_prompt(item_id))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/prompts/{item_id}")
    async def api_save_prompt(item_id: str, request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        try:
            return _dataclass_payload(save_prompt(item_id, str(body.get("body") or "")))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/skills/{item_id}")
    def api_skill(item_id: str) -> dict[str, Any]:
        try:
            return _dataclass_payload(get_skill(item_id))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/skills/{item_id}")
    async def api_save_skill(item_id: str, request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        try:
            return _dataclass_payload(save_skill(item_id, str(body.get("body") or "")))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/experiments/{item_id}")
    def api_experiment(item_id: str) -> dict[str, Any]:
        try:
            experiment = get_experiment(item_id)
            return {
                "experiment": experiment.to_dict(),
                "body": get_experiment_text(item_id),
            }
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/experiments/{item_id}")
    async def api_save_experiment(item_id: str, request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        try:
            if isinstance(body.get("experiment"), dict):
                experiment = save_experiment_data(item_id, body["experiment"])
            else:
                experiment = save_experiment(item_id, str(body.get("body") or ""))
            return {
                "experiment": experiment.to_dict(),
                "body": experiment_to_yaml(experiment),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/preview")
    async def api_preview(request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        try:
            return _render_payload(body, write=False)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/render")
    async def api_render(request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        try:
            return _render_payload(body, write=True)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/preflight")
    async def api_preflight(request: Request) -> JSONResponse:
        body = await _json_body(request)
        command = _cli_command("preflight", _model_args(body), "--no-bridge-up")
        return JSONResponse(start_job("preflight", command))

    @app.post("/api/bridge/up")
    async def api_bridge_up(request: Request) -> JSONResponse:
        body = await _json_body(request)
        command = _cli_command("bridge", "up", _model_args(body))
        return JSONResponse(start_job("bridge-up", command))

    @app.post("/api/prepare")
    async def api_prepare(request: Request) -> JSONResponse:
        body = await _json_body(request)
        _save_body_experiment(body)
        command = _cli_command(
            "prepare",
            *_str_arg("--experiment", body.get("experiment_id")),
            "--manifest",
            str(body.get("manifest") or "datasets/pilot.yaml"),
            *_csv_arg("--memory-variants", body.get("memory_variants")),
        )
        return JSONResponse(start_job("prepare", command))

    @app.post("/api/run")
    async def api_run(request: Request) -> JSONResponse:
        body = await _json_body(request)
        _save_body_experiment(body)
        command = _cli_command(
            "run",
            *_str_arg("--experiment", body.get("experiment_id")),
            "--manifest",
            str(body.get("manifest") or "datasets/pilot.yaml"),
            _model_args(body),
            *_str_arg("--run-name", _run_name_from_body(body)),
            *_str_arg("--tags", body.get("tags")),
            *_csv_arg("--harnesses", body.get("harnesses")),
            *_csv_arg("--variants", body.get("variant_ids")),
            *_int_arg("-l", body.get("n_tasks")),
            *_int_arg("-k", body.get("n_attempts")),
            *_int_arg("-n", body.get("n_concurrent")),
            *(["--dry-run"] if body.get("dry_run", True) else []),
        )
        return JSONResponse(start_job("run", command))

    @app.post("/api/export")
    async def api_export(request: Request) -> JSONResponse:
        body = await _json_body(request)
        jobs = body.get("jobs") or ["jobs/pilot"]
        out = body.get("out") or "reports/pilot.jsonl"
        command = _cli_command("export", "--jobs", *jobs, "--out", out)
        return JSONResponse(start_job("export", command))

    @app.get("/api/jobs")
    def api_jobs() -> list[dict[str, Any]]:
        return list_jobs()

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str) -> dict[str, Any]:
        detail = job_detail(job_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="job not found")
        return detail

    @app.get("/api/jobs/{job_id}/events")
    def api_job_events(job_id: str) -> StreamingResponse:
        if job_detail(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        return StreamingResponse(tail_job_events(job_id), media_type="text/event-stream")

    @app.get("/api/results")
    def api_results(path: str = "jobs") -> dict[str, Any]:
        env = _load_env(Path(".env"))
        rows = _safe_export_rows(Path(path))
        rows = _add_weave_urls(rows, env)
        return {"rows": rows, "summary": _summarize_rows(rows)}

    return app


async def _json_body(request: Any) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def _cli_command(*parts: Any) -> list[str]:
    flat = [str(part) for value in parts for part in _flatten(value) if str(part)]
    return [sys.executable, "-m", "fugue.bench.cli", *flat]


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _model_args(body: dict[str, Any]) -> list[str]:
    return ["--model", str(body["model"])] if body.get("model") else []


def _csv_arg(flag: str, value: Any) -> list[str]:
    values = _coerce_list(value)
    return [flag, ",".join(values)] if values else []


def _int_arg(flag: str, value: Any) -> list[str]:
    return [flag, str(int(value))] if value not in (None, "") else []


def _str_arg(flag: str, value: Any) -> list[str]:
    text = str(value).strip() if value not in (None, "") else ""
    return [flag, text] if text else []


def _coerce_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _run_name_from_body(body: dict[str, Any]) -> str | None:
    value = body.get("run_name")
    if value not in (None, ""):
        return str(value)
    experiment = body.get("experiment")
    if isinstance(experiment, dict):
        return str(
            experiment.get("run_name")
            or experiment.get("title")
            or experiment.get("id")
            or ""
        )
    return None


def _status_payload() -> dict[str, Any]:
    env = _load_env(Path(".env"))
    model = select_model(env=env)
    trace_project = trace_project_slug(env)
    urls = _wandb_urls(trace_project, env)
    try:
        route = resolve_model_route(model, env)
        route_data: dict[str, Any] = {
            "provider": route.provider,
            "model": route.display_model,
            "api_key_env": route.api_key_env,
        }
    except ValueError as exc:
        route_data = {"error": str(exc), "model": model}
    return {
        "route": route_data,
        "default_model": DEFAULT_MODEL,
        "keys": env_presence(
            [
                "WANDB_API_KEY",
                "WANDB_ENTITY",
                "WANDB_PROJECT",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "LITELLM_MASTER_KEY",
            ],
            env,
        ),
        "bridge": bridge_status(),
        "cwd": Path.cwd().as_posix(),
        "weave_project": trace_project,
        "trace_project": trace_project,
        "wandb_app_base_url": _wandb_app_base_url(env),
        "wandb_project_url": urls.get("project"),
        "weave_project_url": urls.get("weave"),
    }


def _manifest_payload(path: str | Path) -> dict[str, Any]:
    manifest = load_manifest(Path(path))
    return {
        "dataset": manifest.dataset.__dict__,
        "model": manifest.model,
        "memory_variants": manifest.memory_variants,
        "harnesses": [harness.__dict__ for harness in manifest.harnesses],
        "tasks": [task.__dict__ for task in manifest.tasks],
        "k": manifest.k,
        "n_concurrent": manifest.n_concurrent,
        "jobs_dir": manifest.jobs_dir.as_posix(),
        "artifact_root": manifest.artifact_root.as_posix(),
        "counts": {
            "tasks": len(manifest.tasks),
            "harnesses": len(manifest.harnesses),
            "memory_variants": len(manifest.memory_variants),
        },
    }


def _library_payload() -> dict[str, Any]:
    return {
        "prompts": [_dataclass_payload(item) for item in list_prompts()],
        "skills": [_dataclass_payload(item) for item in list_skills()],
        "experiments": [_dataclass_payload(item) for item in list_experiments()],
    }


def _dataclass_payload(value: Any) -> dict[str, Any]:
    data = value.__dict__.copy()
    return {key: _jsonable(item) for key, item in data.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _experiment_from_body(body: dict[str, Any], experiment_id: str) -> ExperimentSpec:
    if isinstance(body.get("experiment"), dict):
        data = dict(body["experiment"])
        data["id"] = experiment_id
        return experiment_from_data(data, item_id=experiment_id)
    return get_experiment(experiment_id)


def _save_body_experiment(body: dict[str, Any]) -> None:
    experiment_id = str(body.get("experiment_id") or "")
    if experiment_id and isinstance(body.get("experiment"), dict):
        save_experiment_data(experiment_id, body["experiment"])


def _render_payload(body: dict[str, Any], *, write: bool) -> dict[str, Any]:
    experiment_id = str(body.get("experiment_id") or "pilot")
    experiment = _experiment_from_body(body, experiment_id)
    variants = _variant_override(experiment, body.get("variant_ids"))
    experiment = experiment_with_overrides(
        experiment,
        model=body.get("model"),
        run_name=_run_name_from_body(body),
        tags=_coerce_list(body.get("tags")),
        harnesses=_coerce_list(body.get("harnesses")),
        variants=[variant.to_dict() for variant in variants] if variants else None,
        n_tasks=body.get("n_tasks"),
        n_attempts=body.get("n_attempts"),
        n_concurrent=body.get("n_concurrent"),
    )
    env = _load_env(Path(".env"))
    manifest_path = Path(body.get("manifest") or experiment.manifest)
    manifest = load_manifest(manifest_path)
    renderer = render_jobs if write else preview_jobs
    rendered = renderer(
        experiment=experiment,
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=Path.cwd(),
        env=env,
        model=body.get("model"),
        harness_names=_coerce_list(body.get("harnesses")) or None,
        n_tasks=_optional_int(body.get("n_tasks")),
        n_attempts=_optional_int(body.get("n_attempts")),
        n_concurrent=_optional_int(body.get("n_concurrent")),
        run_name=_run_name_from_body(body),
        tags=_coerce_list(body.get("tags")),
        run_id=_render_run_id(body, write=write),
    )
    return {
        "experiment": experiment.to_dict(),
        "summary": _render_summary(rendered, manifest, experiment),
        "commands": [_rendered_job_payload(job) for job in rendered],
    }


def _variant_override(
    experiment: ExperimentSpec, variant_ids: Any
) -> list[Any] | None:
    selected_ids = set(_coerce_list(variant_ids))
    if not selected_ids:
        return None
    variants = [variant for variant in experiment.variants if variant.id in selected_ids]
    missing = sorted(selected_ids - {variant.id for variant in variants})
    if missing:
        raise ValueError(f"unknown variant(s): {', '.join(missing)}")
    return variants


def _render_run_id(body: dict[str, Any], *, write: bool) -> str:
    if not write:
        return "web-preview"
    return _slug(_run_name_from_body(body) or body.get("experiment_id") or "web")


def _render_summary(
    rendered: list[RenderedJob], manifest: Any, experiment: ExperimentSpec
) -> dict[str, Any]:
    task_count = experiment.n_tasks or len(manifest.tasks)
    trials_per_cell = experiment.n_attempts or manifest.k
    return {
        "cells": len(rendered),
        "task_count": task_count,
        "trials_per_cell": trials_per_cell,
        "estimated_trials": len(rendered) * task_count * trials_per_cell,
        "variants": len({job.variant_id for job in rendered}),
        "harnesses": len({job.harness for job in rendered}),
    }


def _rendered_job_payload(job: RenderedJob) -> dict[str, Any]:
    return {
        "command": job.command,
        "config_path": job.config_path.as_posix(),
        "job_name": job.job_name,
        "harness": job.harness,
        "feature_memory": job.feature_memory,
        "prompt_id": job.prompt_id,
        "skill_ids": job.skill_ids,
        "variant_id": job.variant_id,
        "variant_label": job.variant_label,
        "agent_config_hash": job.agent_config_hash,
        "provider": job.route.provider,
        "model": job.route.display_model,
        "config": job.config,
    }


def _safe_export_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return export_rows([path])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _add_weave_urls(rows: list[dict[str, Any]], env: dict[str, str]) -> list[dict[str, Any]]:
    base = _wandb_app_base_url(env)
    updated = []
    for row in rows:
        item = dict(row)
        if item.get("trace_project"):
            item["weave_url"] = f"{base}/{item['trace_project']}/weave"
        updated.append(item)
    return updated


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("pass") is True)
    failed = sum(1 for row in rows if row.get("pass") is False)
    exceptions = sum(1 for row in rows if row.get("exception_class"))
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "exceptions": exceptions,
        "pass_rate": passed / total if total else None,
        "cost_usd": sum(float(row.get("cost_usd") or 0) for row in rows),
        "tokens": {
            "input": sum(int(row.get("n_input_tokens") or 0) for row in rows),
            "cache": sum(int(row.get("n_cache_tokens") or 0) for row in rows),
            "output": sum(int(row.get("n_output_tokens") or 0) for row in rows),
        },
        "by_experiment_id": _group_rows(rows, "experiment_id"),
        "by_run_name": _group_rows(rows, "run_name"),
        "by_variant_id": _group_rows(rows, "variant_id"),
        "by_prompt": _group_rows(rows, "prompt_id"),
        "by_skill": _group_list_rows(rows, "skill_ids"),
        "by_feature_memory": _group_rows(rows, "feature_memory"),
        "by_harness": _group_rows(rows, "harness"),
        "by_provider": _group_rows(rows, "model_provider"),
        "latest_failure_count": failed + exceptions,
    }


def _group_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    grouped = []
    for name, group in sorted(groups.items()):
        passed = sum(1 for row in group if row.get("pass") is True)
        total = len(group)
        grouped.append(
            {
                "name": name,
                "total": total,
                "passed": passed,
                "failed": sum(1 for row in group if row.get("pass") is False),
                "pass_rate": passed / total if total else None,
                "cost_usd": sum(float(row.get("cost_usd") or 0) for row in group),
            }
        )
    return grouped


def _group_list_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        values = row.get(key) or ["none"]
        if not isinstance(values, list):
            values = [values]
        for value in values or ["none"]:
            item = dict(row)
            item[key] = value
            expanded.append(item)
    return _group_rows(expanded, key)


def _wandb_urls(trace_project: str | None, env: dict[str, str]) -> dict[str, str | None]:
    if not trace_project:
        return {"project": None, "weave": None}
    base = _wandb_app_base_url(env)
    return {
        "project": f"{base}/{trace_project}",
        "weave": f"{base}/{trace_project}/weave",
    }


def _wandb_app_base_url(env: dict[str, str]) -> str:
    return env.get("WANDB_APP_BASE_URL", "https://wandb.ai").rstrip("/")


def _model_key_ready(status: dict[str, Any]) -> bool:
    key = (status.get("route") or {}).get("api_key_env")
    return bool(key and (status.get("keys") or {}).get(key))


def _slug(value: Any) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value).strip())
    return "-".join(part for part in out.split("-") if part) or "web"
