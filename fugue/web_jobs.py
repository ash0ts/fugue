from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from fugue.bench.cli import _load_env
from fugue.redaction import redact_text, secrets_from_env

WEB_JOBS_DIR = Path("jobs") / "web"
_SERVER_ID = uuid.uuid4().hex
_META_LOCK = Lock()
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def start_job(kind: str, command: list[str]) -> dict[str, Any]:
    WEB_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    job_id = f"{stamp}-{_slug(kind)}-{uuid.uuid4().hex[:8]}"
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
        "command": _redact_command(command),
        "pid": process.pid,
        "owner": _SERVER_ID,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "log_path": log_path.as_posix(),
    }
    _write_json(meta_path, meta)

    def wait_for_process() -> None:
        returncode = process.wait()
        log_file.close()
        current = _read_json(meta_path) or meta
        current["status"] = "succeeded" if returncode == 0 else "failed"
        current["returncode"] = returncode
        current["ended_at"] = datetime.now(UTC).isoformat()
        _write_json(meta_path, current)

    Thread(target=wait_for_process, daemon=True).start()
    return meta


def list_jobs() -> list[dict[str, Any]]:
    if not WEB_JOBS_DIR.exists():
        return []
    jobs = []
    for meta_path in sorted(WEB_JOBS_DIR.glob("*/meta.json"), reverse=True):
        meta = _read_json(meta_path)
        if meta:
            jobs.append(_recover_meta(meta_path, meta))
    return jobs


def job_detail(job_id: str) -> dict[str, Any] | None:
    job_dir = _job_dir(job_id)
    if job_dir is None:
        return None
    meta_path = job_dir / "meta.json"
    meta = _read_json(meta_path)
    if meta is None:
        return None
    meta = _recover_meta(meta_path, meta)
    log_path = job_dir / "output.log"
    meta["log_tail"] = _sanitize_log(_read_tail(log_path, 8_000))
    return meta


def tail_job_events(job_id: str):
    job_dir = _job_dir(job_id)
    if job_dir is None:
        return
    log_path = job_dir / "output.log"
    meta_path = job_dir / "meta.json"
    position = 0
    while True:
        if log_path.exists():
            with log_path.open("rb") as handle:
                handle.seek(position)
                chunk = handle.read(64 * 1024)
                position = handle.tell()
            if chunk:
                text = _sanitize_log(chunk.decode(errors="replace"))
                yield f"data: {json.dumps({'chunk': text})}\n\n"
        status = _read_json(meta_path) or {}
        if status:
            status = _recover_meta(meta_path, status)
        if status.get("status") != "running":
            yield f"data: {json.dumps({'done': True, 'status': status})}\n\n"
            break
        time.sleep(0.5)


def _recover_meta(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    if meta.get("status") != "running":
        return meta
    pid = meta.get("pid")
    owned = meta.get("owner") == _SERVER_ID
    if owned and isinstance(pid, int) and _pid_alive(pid):
        return meta
    recovered = dict(meta)
    recovered["status"] = "interrupted"
    recovered["ended_at"] = datetime.now(UTC).isoformat()
    recovered["error"] = "Web server restarted or the managed process exited unexpectedly."
    _write_json(path, recovered)
    return recovered


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with _META_LOCK:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)


def _read_tail(path: Path, size: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            length = handle.tell()
            handle.seek(max(0, length - size))
            return handle.read().decode(errors="replace")
    except OSError:
        return ""


def _sanitize_log(value: str) -> str:
    env = _load_env(Path(".env"))
    return redact_text(value, secrets_from_env(env))


def _job_dir(job_id: str) -> Path | None:
    if not _JOB_ID.fullmatch(job_id):
        return None
    return WEB_JOBS_DIR / job_id


def _redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for value in command:
        if hide_next:
            redacted.append("[redacted]")
            hide_next = False
            continue
        redacted.append(value)
        hide_next = value.lower() in {
            "--api-key",
            "--token",
            "--password",
            "--secret",
        }
    return redacted


def _slug(value: str) -> str:
    selected = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in selected.split("-") if part) or "job"


def _prepend_cwd(existing: str | None) -> str:
    cwd = Path.cwd().as_posix()
    return cwd if not existing else f"{cwd}:{existing}"
