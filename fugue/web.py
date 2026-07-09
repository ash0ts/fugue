from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Any

from fugue.bench.cli import _load_env
from fugue.bench.export import export_rows
from fugue.bench.manifest import load_manifest
from fugue.bridge import bridge_status
from fugue.model_plane import (
    DEFAULT_MODEL,
    env_presence,
    resolve_model_route,
    select_model,
)

WEB_JOBS_DIR = Path("jobs") / "web"
STATIC_DIR = Path(__file__).resolve().parent / "web_static"


def run_web(host: str = "127.0.0.1", port: int = 8765) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('install web dependencies with: uv pip install -e ".[web]"') from exc

    uvicorn.run("fugue.web:create_app", factory=True, host=host, port=port)


def create_app():
    try:
        from fastapi import FastAPI, HTTPException, Request
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
        jobs = _list_jobs()
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
            "matrix": _matrix(rows, manifest),
        }

    @app.get("/api/manifest")
    def api_manifest(path: str = "datasets/pilot.yaml") -> dict[str, Any]:
        return _manifest_payload(path)

    @app.post("/api/preflight")
    async def api_preflight(request: Request) -> JSONResponse:
        body = await _json_body(request)
        command = _cli_command("preflight", _model_args(body), "--no-bridge-up")
        return JSONResponse(_start_job("preflight", command))

    @app.post("/api/bridge/up")
    async def api_bridge_up(request: Request) -> JSONResponse:
        body = await _json_body(request)
        command = _cli_command("bridge", "up", _model_args(body))
        return JSONResponse(_start_job("bridge-up", command))

    @app.post("/api/prepare")
    async def api_prepare(request: Request) -> JSONResponse:
        body = await _json_body(request)
        command = _cli_command(
            "prepare",
            "--manifest",
            str(body.get("manifest") or "datasets/pilot.yaml"),
            *_csv_arg("--conditions", body.get("conditions")),
        )
        return JSONResponse(_start_job("prepare", command))

    @app.post("/api/run")
    async def api_run(request: Request) -> JSONResponse:
        body = await _json_body(request)
        command = _cli_command(
            "run",
            "--manifest",
            str(body.get("manifest") or "datasets/pilot.yaml"),
            _model_args(body),
            *_csv_arg("--harnesses", body.get("harnesses")),
            *_csv_arg("--conditions", body.get("conditions")),
            *_int_arg("-l", body.get("n_tasks")),
            *_int_arg("-k", body.get("n_attempts")),
            *_int_arg("-n", body.get("n_concurrent")),
            *(["--dry-run"] if body.get("dry_run", True) else []),
        )
        return JSONResponse(_start_job("run", command))

    @app.post("/api/export")
    async def api_export(request: Request) -> JSONResponse:
        body = await _json_body(request)
        jobs = body.get("jobs") or ["jobs/pilot"]
        out = body.get("out") or "reports/pilot.jsonl"
        command = _cli_command("export", "--jobs", *jobs, "--out", out)
        return JSONResponse(_start_job("export", command))

    @app.get("/api/jobs")
    def api_jobs() -> list[dict[str, Any]]:
        return _list_jobs()

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str) -> dict[str, Any]:
        job_dir = WEB_JOBS_DIR / job_id
        meta_path = job_dir / "meta.json"
        if not meta_path.is_file():
            raise HTTPException(status_code=404, detail="job not found")
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500, detail="job metadata is invalid"
            ) from exc
        log_path = job_dir / "output.log"
        if log_path.exists():
            meta["log_tail"] = log_path.read_text(errors="replace")[-8000:]
        else:
            meta["log_tail"] = ""
        return meta

    @app.get("/api/jobs/{job_id}/events")
    def api_job_events(job_id: str) -> StreamingResponse:
        job_dir = WEB_JOBS_DIR / job_id
        if not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="job not found")
        return StreamingResponse(_tail_log(job_dir), media_type="text/event-stream")

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
    flat = [str(p) for part in parts for p in _flatten(part) if str(p)]
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
    if not value:
        return []
    if isinstance(value, list):
        value = ",".join(str(item) for item in value if str(item))
    return [flag, str(value)]


def _int_arg(flag: str, value: Any) -> list[str]:
    return [flag, str(int(value))] if value not in (None, "") else []


def _start_job(kind: str, command: list[str]) -> dict[str, Any]:
    WEB_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    job_id = f"{stamp}-{kind}"
    job_dir = WEB_JOBS_DIR / job_id
    suffix = 1
    while job_dir.exists():
        suffix += 1
        job_id = f"{stamp}-{kind}-{suffix}"
        job_dir = WEB_JOBS_DIR / job_id
    job_dir.mkdir()
    log_path = job_dir / "output.log"
    meta_path = job_dir / "meta.json"
    env = _load_env(Path(".env"))
    env["PYTHONPATH"] = _prepend_cwd(env.get("PYTHONPATH"))
    log_file = log_path.open("w")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=Path.cwd(),
    )
    meta = {
        "id": job_id,
        "kind": kind,
        "command": command,
        "pid": process.pid,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "log_path": log_path.as_posix(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    def wait_for_process() -> None:
        returncode = process.wait()
        log_file.close()
        current = json.loads(meta_path.read_text())
        current["status"] = "succeeded" if returncode == 0 else "failed"
        current["returncode"] = returncode
        current["ended_at"] = datetime.now(UTC).isoformat()
        meta_path.write_text(json.dumps(current, indent=2) + "\n")

    Thread(target=wait_for_process, daemon=True).start()
    return meta


def _list_jobs() -> list[dict[str, Any]]:
    if not WEB_JOBS_DIR.exists():
        return []
    jobs = []
    for meta_path in sorted(WEB_JOBS_DIR.glob("*/meta.json"), reverse=True):
        try:
            jobs.append(json.loads(meta_path.read_text()))
        except json.JSONDecodeError:
            continue
    return jobs


def _status_payload() -> dict[str, Any]:
    env = _load_env(Path(".env"))
    model = select_model(env=env)
    trace_project = _trace_project(env)
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
        "conditions": manifest.conditions,
        "harnesses": [h.__dict__ for h in manifest.harnesses],
        "tasks": [t.__dict__ for t in manifest.tasks],
        "k": manifest.k,
        "n_concurrent": manifest.n_concurrent,
        "jobs_dir": manifest.jobs_dir.as_posix(),
        "artifact_root": manifest.artifact_root.as_posix(),
        "counts": {
            "tasks": len(manifest.tasks),
            "harnesses": len(manifest.harnesses),
            "conditions": len(manifest.conditions),
            "matrix_cells": len(manifest.harnesses) * len(manifest.conditions),
        },
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


def _tail_log(job_dir: Path):
    log_path = job_dir / "output.log"
    meta_path = job_dir / "meta.json"
    position = 0
    while True:
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            if len(text) > position:
                chunk = text[position:]
                position = len(text)
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        status = {}
        if meta_path.exists():
            status = json.loads(meta_path.read_text())
        if status.get("status") != "running":
            yield f"data: {json.dumps({'done': True, 'status': status})}\n\n"
            break
        time.sleep(1)


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
        "by_harness": _group_rows(rows, "harness"),
        "by_condition": _group_rows(rows, "condition"),
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


def _matrix(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("harness") or ""), str(row.get("condition") or "none"))
        by_cell.setdefault(key, []).append(row)
    matrix = []
    for harness in manifest.get("harnesses", []):
        harness_name = harness["name"]
        cells = []
        for condition in manifest.get("conditions", []):
            group = by_cell.get((harness_name, condition), [])
            cells.append(
                {
                    "condition": condition,
                    "status": _cell_status(group),
                    "total": len(group),
                    "passed": sum(1 for row in group if row.get("pass") is True),
                }
            )
        matrix.append({"harness": harness_name, "cells": cells})
    return matrix


def _cell_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "ready"
    if any(row.get("pass") is False or row.get("exception_class") for row in rows):
        return "failed"
    if any(row.get("pass") is True for row in rows):
        return "passed"
    return "not run"


def _trace_project(env: dict[str, str]) -> str | None:
    weave_project = env.get("WEAVE_PROJECT", "").strip()
    if "/" in weave_project:
        return weave_project
    entity = env.get("WANDB_ENTITY", "").strip()
    project = env.get("WANDB_PROJECT", "").strip()
    if entity and project:
        return f"{entity}/{project}"
    return None


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


def _prepend_cwd(existing: str | None) -> str:
    cwd = Path.cwd().as_posix()
    return cwd if not existing else f"{cwd}:{existing}"
