from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Any

from fugue.bench.cli import _load_env

WEB_JOBS_DIR = Path("jobs") / "web"


def start_job(kind: str, command: list[str]) -> dict[str, Any]:
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


def list_jobs() -> list[dict[str, Any]]:
    if not WEB_JOBS_DIR.exists():
        return []
    jobs = []
    for meta_path in sorted(WEB_JOBS_DIR.glob("*/meta.json"), reverse=True):
        try:
            jobs.append(json.loads(meta_path.read_text()))
        except json.JSONDecodeError:
            continue
    return jobs


def job_detail(job_id: str) -> dict[str, Any] | None:
    meta_path = WEB_JOBS_DIR / job_id / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    log_path = WEB_JOBS_DIR / job_id / "output.log"
    meta["log_tail"] = log_path.read_text(errors="replace")[-8000:] if log_path.exists() else ""
    return meta


def tail_job_events(job_id: str):
    job_dir = WEB_JOBS_DIR / job_id
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


def _prepend_cwd(existing: str | None) -> str:
    cwd = Path.cwd().as_posix()
    return cwd if not existing else f"{cwd}:{existing}"
