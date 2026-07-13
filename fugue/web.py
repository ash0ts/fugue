from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from fastapi import Request
except ImportError:  # pragma: no cover - web extra not installed
    Request = Any  # type: ignore

from fugue.bench.cli import (
    _direct_workload_jobs,
    _load_env,
    _preset_workload_int,
    _selected_preset,
    _selected_system_ids,
    _selected_workloads,
)
from fugue.bench.context import (
    DEFAULT_CACHE_ROOT,
    ContextRuntime,
    list_context_systems,
    preflight_context,
    run_async,
)
from fugue.bench.execution import new_run_id
from fugue.bench.export import export_rows
from fugue.bench.job_config import RenderedJob, preview_jobs, render_jobs
from fugue.bench.library import (
    ExperimentSpec,
    WorkloadSpec,
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
from fugue.bench.scoring import pareto_frontier, summarize_metric_rows
from fugue.bridge import bridge_status
from fugue.model_plane import (
    DEFAULT_MODEL,
    env_presence,
    resolve_model_route,
    select_model,
    trace_project_slug,
)
from fugue.redaction import redact_value, secrets_from_env
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
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:
        raise RuntimeError('install web dependencies with: uv pip install -e ".[web]"') from exc

    app = FastAPI(title="Fugue")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def disable_local_asset_cache(request: Request, call_next: Any):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def api_status(
        model: str | None = None,
        builder_model: str | None = None,
        judge_model: str | None = None,
        include_context: bool = False,
    ) -> dict[str, Any]:
        return _status_payload(
            model, builder_model, judge_model, include_context=include_context
        )

    @app.get("/api/summary")
    def api_summary(
        experiment_id: str | None = None,
        model: str | None = None,
        builder_model: str | None = None,
        judge_model: str | None = None,
    ) -> dict[str, Any]:
        experiment, manifest_path = _active_experiment_manifest(experiment_id)
        status = _status_payload(
            model or (experiment.model if experiment else None),
            builder_model or (experiment.builder_model if experiment else None),
            judge_model or (experiment.judge_model if experiment else None),
            include_context=True,
        )
        if experiment:
            preset = _selected_preset(experiment, None)
            active_systems = set(
                preset.systems
                or [
                    variant.context.system_id
                    for variant in experiment.variants
                    if variant.enabled
                ]
            )
            status["context_systems"] = [
                item
                for item in status["context_systems"]
                if item["id"] in active_systems
            ]
        manifest = _manifest_payload(manifest_path)
        rows = _safe_export_rows(Path("jobs"))
        jobs = list_jobs()
        result_summary = _summarize_rows(rows)
        readiness = {
            "trace": bool(status.get("weave_project"))
            and status["keys"].get("WANDB_API_KEY", False),
            "model": _model_key_ready(status, "target"),
            "builder": _model_key_ready(status, "builder"),
            "judge": _model_key_ready(status, "judge"),
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

    @app.get("/api/context-systems")
    def api_context_systems() -> list[dict[str, Any]]:
        return [_context_system_payload(spec) for spec in list_context_systems()]

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
            return await run_in_threadpool(_render_payload, body, write=False)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/render")
    async def api_render(request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        try:
            return await run_in_threadpool(_render_payload, body, write=True)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/preflight")
    async def api_preflight(request: Request) -> JSONResponse:
        body = await _json_body(request)
        command = _cli_command(
            "preflight",
            _model_args(body),
            *_str_arg("--builder-model", body.get("builder_model")),
            *_str_arg("--judge-model", body.get("judge_model")),
            *_str_arg("--experiment", body.get("experiment_id")),
            *_str_arg("--preset", body.get("preset")),
            *_csv_arg("--systems", body.get("systems")),
            "--no-bridge-up",
        )
        return JSONResponse(start_job("preflight", command))

    @app.post("/api/bridge/up")
    async def api_bridge_up(request: Request) -> JSONResponse:
        body = await _json_body(request)
        command = _cli_command(
            "bridge",
            "up",
            _model_args(body),
            *_str_arg("--builder-model", body.get("builder_model")),
            *_str_arg("--judge-model", body.get("judge_model")),
        )
        return JSONResponse(start_job("bridge-up", command))

    @app.post("/api/prepare")
    async def api_prepare(request: Request) -> JSONResponse:
        body = await _json_body(request)
        command = _cli_command(
            "context",
            "prepare",
            *_str_arg("--experiment", body.get("experiment_id")),
            *_str_arg("--manifest", body.get("manifest")),
            *_str_arg("--preset", body.get("preset")),
            *_csv_arg("--workloads", body.get("workloads")),
            *_csv_arg("--systems", body.get("systems")),
            *_str_arg("--model", body.get("model")),
            *_str_arg("--builder-model", body.get("builder_model")),
            *(["--rebuild"] if body.get("rebuild_context") else []),
        )
        return JSONResponse(start_job("prepare", command))

    @app.post("/api/run")
    async def api_run(request: Request) -> JSONResponse:
        body = await _json_body(request)
        command = _cli_command(
            "run",
            *_str_arg("--experiment", body.get("experiment_id")),
            *_str_arg("--manifest", body.get("manifest")),
            _model_args(body),
            *_str_arg("--builder-model", body.get("builder_model")),
            *_str_arg("--judge-model", body.get("judge_model")),
            *_str_arg("--preset", body.get("preset")),
            *_csv_arg("--workloads", body.get("workloads")),
            *_csv_arg("--systems", body.get("systems")),
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
        jobs = _coerce_list(body.get("jobs")) or ["jobs", ".fugue/runtime"]
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
    except (ValueError, UnicodeDecodeError):
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


def _status_payload(
    model: str | None = None,
    builder_model: str | None = None,
    judge_model: str | None = None,
    *,
    include_context: bool = False,
) -> dict[str, Any]:
    env = _load_env(Path(".env"))
    model = select_model(model, env=env)
    builder_model = builder_model or env.get("FUGUE_BUILDER_MODEL") or model
    judge_model = judge_model or env.get("FUGUE_JUDGE_MODEL")
    trace_project = trace_project_slug(env)
    urls = _wandb_urls(trace_project, env)
    routes = {
        "target": _route_payload(model, env),
        "builder": _route_payload(builder_model, env),
        "judge": _route_payload(judge_model, env) if judge_model else None,
    }
    return {
        "route": routes["target"],
        "routes": routes,
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
        "bridge": bridge_status(timeout_sec=0.5),
        "cwd": Path.cwd().as_posix(),
        "weave_project": trace_project,
        "trace_project": trace_project,
        "wandb_app_base_url": _wandb_app_base_url(env),
        "wandb_project_url": urls.get("project"),
        "weave_project_url": urls.get("weave"),
        "judge_model": judge_model,
        "builder_model": builder_model,
        "context_cache_root": DEFAULT_CACHE_ROOT.as_posix(),
        "context_systems": (
            [
                _context_system_payload(spec, env=env)
                for spec in list_context_systems()
            ]
            if include_context
            else []
        ),
    }


def _route_payload(model: str, env: dict[str, str]) -> dict[str, Any]:
    try:
        route = resolve_model_route(model, env)
        return {
            "provider": route.provider,
            "model": route.display_model,
            "api_key_env": route.api_key_env,
        }
    except ValueError as exc:
        return {"error": str(exc), "model": model}


def _active_experiment_manifest(
    experiment_id: str | None,
) -> tuple[ExperimentSpec | None, Path]:
    if experiment_id:
        try:
            experiment = get_experiment(experiment_id)
            preset = _selected_preset(experiment, None)
            workload = next(
                (
                    item
                    for item in _selected_workloads(experiment, preset, None)
                    if item.runner == "harbor" and item.manifest
                ),
                None,
            )
            return experiment, Path(
                workload.manifest if workload else experiment.manifest
            )
        except FileNotFoundError:
            pass
    return None, Path("datasets/pilot.yaml")


def _manifest_payload(path: str | Path) -> dict[str, Any]:
    manifest = load_manifest(Path(path))
    return {
        "dataset": manifest.dataset.__dict__,
        "model": manifest.model,
        "harnesses": [harness.__dict__ for harness in manifest.harnesses],
        "tasks": [task.__dict__ for task in manifest.tasks],
        "k": manifest.k,
        "n_concurrent": manifest.n_concurrent,
        "jobs_dir": manifest.jobs_dir.as_posix(),
        "counts": {
            "tasks": len(manifest.tasks),
            "harnesses": len(manifest.harnesses),
        },
    }


def _library_payload() -> dict[str, Any]:
    return {
        "prompts": [_dataclass_payload(item) for item in list_prompts()],
        "skills": [_dataclass_payload(item) for item in list_skills()],
        "experiments": [_dataclass_payload(item) for item in list_experiments()],
    }


def _context_system_payload(spec: Any, env: dict[str, str] | None = None) -> dict[str, Any]:
    values = env or _load_env(Path(".env"))
    runtime = ContextRuntime(
        repo_root=Path.cwd(),
        cache_root=Path.cwd() / DEFAULT_CACHE_ROOT,
        env=values,
    )
    checks = run_async(preflight_context(spec, runtime))
    license_check = next((item for item in checks if item.name == "license"), None)
    return {
        "id": spec.id,
        "title": spec.title,
        "description": spec.description,
        "version": spec.version,
        "capabilities": sorted(spec.capabilities),
        "license": spec.license,
        "license_url": spec.license_url,
        "enabled_by_default": spec.enabled_by_default,
        "requires_license_approval": spec.requires_license_approval,
        "license_ready": license_check.ok if license_check else True,
        "checks": [
            {
                "name": item.name,
                "ok": item.ok,
                "detail": item.detail,
                "severity": item.severity,
                "phase": item.phase,
            }
            for item in checks
        ],
        "ready": all(item.ok or item.severity == "warning" for item in checks),
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


def _render_payload(body: dict[str, Any], *, write: bool) -> dict[str, Any]:
    experiment_id = str(body.get("experiment_id") or "pilot")
    experiment = _experiment_from_body(body, experiment_id)
    variants = _variant_override(experiment, body.get("variant_ids"))
    experiment = experiment_with_overrides(
        experiment,
        model=body.get("model"),
        builder_model=body.get("builder_model"),
        judge_model=body.get("judge_model"),
        run_name=_run_name_from_body(body),
        tags=_coerce_list(body.get("tags")),
        harnesses=_coerce_list(body.get("harnesses")),
        variants=[variant.to_dict() for variant in variants] if variants else None,
        n_tasks=body.get("n_tasks"),
        n_attempts=body.get("n_attempts"),
        n_concurrent=body.get("n_concurrent"),
    )
    env = _load_env(Path(".env"))
    env["FUGUE_BUILDER_MODEL"] = (
        experiment.builder_model
        or env.get("FUGUE_BUILDER_MODEL")
        or experiment.model
        or ""
    )
    preset = _selected_preset(experiment, str(body.get("preset") or "") or None)
    workloads = _selected_workloads(
        experiment, preset, _coerce_list(body.get("workloads")) or None
    )
    if not workloads:
        workloads = [
            WorkloadSpec(
                id="harbor",
                runner="harbor",
                manifest=Path(body.get("manifest") or experiment.manifest),
            )
        ]
    rendered: list[RenderedJob] = []
    primary_manifest = None
    run_id = _render_run_id(write=write)
    for workload in workloads:
        if workload.runner == "harbor":
            manifest_path = Path(
                body.get("manifest") or workload.manifest or experiment.manifest
            )
            manifest = load_manifest(manifest_path)
            primary_manifest = primary_manifest or manifest
            renderer = render_jobs if write else preview_jobs
            rendered.extend(
                renderer(
                    experiment=experiment,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    repo_root=Path.cwd(),
                    env=env,
                    model=body.get("model"),
                    harness_names=_coerce_list(body.get("harnesses"))
                    or preset.harnesses
                    or None,
                    system_names=_selected_system_ids(
                        experiment,
                        workload,
                        preset,
                        _coerce_list(body.get("systems")) or None,
                    ),
                    n_tasks=_optional_int(body.get("n_tasks"))
                    or _preset_workload_int(preset, workload.id, "n_tasks")
                    or workload.n_tasks
                    or preset.n_tasks,
                    n_attempts=_optional_int(body.get("n_attempts"))
                    or _preset_workload_int(preset, workload.id, "n_attempts")
                    or workload.n_attempts
                    or preset.n_attempts,
                    n_concurrent=_optional_int(body.get("n_concurrent"))
                    or _preset_workload_int(preset, workload.id, "n_concurrent")
                    or preset.n_concurrent,
                    run_name=_run_name_from_body(body),
                    tags=_coerce_list(body.get("tags")),
                    run_id=run_id,
                    workload_id=workload.id,
                    preset_id=preset.id if preset.id != "default" else None,
                    required_capabilities=workload.required_capabilities,
                    workload_artifacts=workload.artifacts,
                )
            )
        else:
            rendered.extend(
                _direct_workload_jobs(
                    experiment=experiment,
                    workload=workload,
                    preset=preset,
                    env=env,
                    repo_root=Path.cwd(),
                    run_name=_run_name_from_body(body) or experiment.id,
                    model=body.get("model"),
                    requested_systems=_coerce_list(body.get("systems")) or None,
                    n_tasks=_optional_int(body.get("n_tasks")),
                    n_attempts=_optional_int(body.get("n_attempts")),
                    n_concurrent=_optional_int(body.get("n_concurrent")),
                    run_id=run_id,
                )
            )
    return {
        "experiment": experiment.to_dict(),
        "summary": _render_summary(rendered, primary_manifest, experiment),
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


def _render_run_id(*, write: bool) -> str:
    return new_run_id() if write else "web-preview"


def _render_summary(
    rendered: list[RenderedJob], manifest: Any | None, experiment: ExperimentSpec
) -> dict[str, Any]:
    by_workload: dict[str, dict[str, Any]] = {}
    for job in rendered:
        fugue = job.config.get("fugue") or {}
        if job.harness in {"direct", "sequence"}:
            task_count = int(fugue.get("task_count") or 0)
            attempts = int(fugue.get("n_attempts") or 1)
        elif job.applicable:
            datasets = job.config.get("datasets") or []
            task_count = len(
                (datasets[0] if datasets else {}).get("task_names") or []
            )
            attempts = int(job.config.get("n_attempts") or 1)
        else:
            task_count = 0
            attempts = int(job.config.get("n_attempts") or 1)
        item = by_workload.setdefault(
            job.workload_id,
            {
                "workload_id": job.workload_id,
                "cells": 0,
                "applicable_cells": 0,
                "task_count": task_count,
                "trials_per_cell": attempts,
                "estimated_trials": 0,
            },
        )
        item["cells"] += 1
        item["applicable_cells"] += int(job.applicable)
        item["estimated_trials"] += task_count * attempts if job.applicable else 0
    workload_breakdown = list(by_workload.values())
    attempt_values = {item["trials_per_cell"] for item in workload_breakdown}
    direct_runners = sorted(
        {job.harness for job in rendered if job.harness in {"direct", "sequence"}}
    )
    return {
        "cells": len(rendered),
        "task_count": sum(item["task_count"] for item in workload_breakdown),
        "trials_per_cell": attempt_values.pop() if len(attempt_values) == 1 else None,
        "estimated_trials": sum(
            item["estimated_trials"] for item in workload_breakdown
        ),
        "variants": len({job.variant_id for job in rendered}),
        "harnesses": len(
            {job.harness for job in rendered if job.harness not in {"direct", "sequence"}}
        ),
        "direct_runners": direct_runners,
        "workloads": len({job.workload_id for job in rendered}),
        "systems": len({job.context_system_id for job in rendered}),
        "applicable_cells": sum(1 for job in rendered if job.applicable),
        "skipped_cells": sum(1 for job in rendered if not job.applicable),
        "cache_ready_cells": sum(1 for job in rendered if job.context_cache_ready),
        "workload_breakdown": workload_breakdown,
    }


def _rendered_job_payload(job: RenderedJob) -> dict[str, Any]:
    return {
        "command": job.command,
        "config_path": job.config_path.as_posix(),
        "job_name": job.job_name,
        "harness": job.harness,
        "context_system_id": job.context_system_id,
        "context_version": job.context_version,
        "context_cache_keys": job.context_cache_keys,
        "context_cache_ready": job.context_cache_ready,
        "workload_id": job.workload_id,
        "preset_id": job.preset_id,
        "applicable": job.applicable,
        "skip_reason": job.skip_reason,
        "prompt_id": job.prompt_id,
        "skill_ids": job.skill_ids,
        "variant_id": job.variant_id,
        "variant_label": job.variant_label,
        "agent_config_hash": job.agent_config_hash,
        "provider": job.route.provider,
        "model": job.route.display_model,
        "run_id": job.run_id,
        "task_id": job.task_id,
        "config": job.config,
    }


def _safe_export_rows(path: Path) -> list[dict[str, Any]]:
    try:
        sources = [path]
        runtime = Path(".fugue/runtime")
        if runtime.exists() and runtime != path:
            sources.append(runtime)
        env = _load_env(Path(".env"))
        secrets = secrets_from_env(env)
        return [redact_value(row, secrets=secrets) for row in export_rows(sources)]
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
    trial_rows = [row for row in rows if row.get("record_type") == "trial"]
    total = len(trial_rows)
    passed = sum(1 for row in trial_rows if row.get("pass") is True)
    failed = sum(1 for row in trial_rows if row.get("pass") is False)
    scored = sum(1 for row in trial_rows if row.get("pass") is not None)
    exceptions = sum(1 for row in trial_rows if row.get("exception_class"))
    run_ids = [str(row["run_id"]) for row in trial_rows if row.get("run_id")]
    latest_run = max(run_ids) if run_ids else None
    latest_rows = [
        row for row in trial_rows if str(row.get("run_id")) == latest_run
    ]
    latest_failures = sum(
        1
        for row in latest_rows
        if row.get("pass") is False or bool(row.get("exception_class"))
    )
    comparison = summarize_metric_rows(
        rows, ("workload_id", "context_system_id", "harness")
    )
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "scored": scored,
        "exceptions": exceptions,
        "pass_rate": passed / scored if scored else None,
        "judge_overall": _average(trial_rows, "judge_overall"),
        "cost_usd": sum(float(row.get("cost_usd") or 0) for row in trial_rows),
        "tokens": {
            "input": sum(int(row.get("n_input_tokens") or 0) for row in trial_rows),
            "cache": sum(int(row.get("n_cache_tokens") or 0) for row in trial_rows),
            "output": sum(int(row.get("n_output_tokens") or 0) for row in trial_rows),
        },
        "by_experiment_id": _group_rows(trial_rows, "experiment_id"),
        "by_run_name": _group_rows(trial_rows, "run_name"),
        "by_variant_id": _group_rows(trial_rows, "variant_id"),
        "by_prompt": _group_rows(trial_rows, "prompt_id"),
        "by_skill": _group_list_rows(trial_rows, "skill_ids"),
        "by_context_system": _group_rows(trial_rows, "context_system_id"),
        "by_workload": _group_rows(trial_rows, "workload_id"),
        "by_harness": _group_rows(trial_rows, "harness"),
        "by_provider": _group_rows(trial_rows, "model_provider"),
        "comparison": comparison,
        "pareto": pareto_frontier(
            comparison, quality="outcome_quality", cost="wall_time_sec"
        ),
        "latest_failure_count": latest_failures,
    }


def _group_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    grouped = []
    for name, group in sorted(groups.items()):
        passed = sum(1 for row in group if row.get("pass") is True)
        total = len(group)
        scored = sum(1 for row in group if row.get("pass") is not None)
        grouped.append(
            {
                "name": name,
                "total": total,
                "passed": passed,
                "scored": scored,
                "failed": sum(1 for row in group if row.get("pass") is False),
                "pass_rate": passed / scored if scored else None,
                "judge_overall": _average(group, "judge_overall"),
                "cost_usd": sum(float(row.get("cost_usd") or 0) for row in group),
            }
        )
    return grouped


def _average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


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


def _model_key_ready(status: dict[str, Any], role: str) -> bool:
    route = (status.get("routes") or {}).get(role)
    if route is None:
        return True
    key = route.get("api_key_env")
    return bool(key and (status.get("keys") or {}).get(key))
