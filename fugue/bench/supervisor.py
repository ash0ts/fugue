from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fugue.bench.execution import (
    list_run_manifests,
    mark_unfinished_cells,
    read_run_manifest,
    write_run_manifest,
)


@dataclass(frozen=True)
class ManagedRun:
    run_id: str
    status: str
    run_name: str
    experiment_id: str
    pid: int | None
    created_at: str | None
    ended_at: str | None
    run_dir: Path
    log_path: Path
    metadata: dict[str, Any]


class RunSupervisor:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.runtime_root = self.repo_root / ".fugue" / "runtime"

    def start_detached(
        self,
        *,
        run_id: str,
        command: list[str],
        env: dict[str, str],
        run_name: str,
        experiment_id: str,
    ) -> ManagedRun:
        run_dir = self.runtime_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "combined.log"
        write_run_manifest(
            self.repo_root,
            run_id,
            {
                "status": "starting",
                "run_name": run_name,
                "experiment_id": experiment_id,
                "command": _redact_command(command),
                "combined_log": log_path.as_posix(),
                "detached": True,
            },
        )
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        write_run_manifest(
            self.repo_root,
            run_id,
            {
                "pid": process.pid,
                "process_group": process.pid,
            },
        )
        return self.get(run_id, recover=False)

    def list(self, *, recover: bool = True) -> list[ManagedRun]:
        runs: list[ManagedRun] = []
        for metadata in list_run_manifests(self.repo_root):
            run_id = str(metadata.get("run_id") or "")
            if not run_id:
                continue
            if recover:
                metadata = self._recover(run_id, metadata)
            runs.append(self._managed(metadata))
        return runs

    def get(self, run_id: str, *, recover: bool = True) -> ManagedRun:
        run_dir = self._run_dir(run_id)
        metadata = read_run_manifest(run_dir)
        if metadata is None:
            raise FileNotFoundError(f"run not found: {run_id}")
        if recover:
            metadata = self._recover(run_id, metadata)
        return self._managed(metadata)

    def cancel(self, run_id: str) -> ManagedRun:
        run = self.get(run_id)
        if run.status not in {"starting", "running"}:
            return run
        process_group = run.metadata.get("process_group") or run.pid
        if isinstance(process_group, int):
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
        message = "Run cancelled by the operator."
        mark_unfinished_cells(run.run_dir, "cancelled", message=message)
        write_run_manifest(
            self.repo_root,
            run_id,
            {
                "status": "cancelled",
                "ended_at": _now(),
                "error": message,
            },
        )
        return self.get(run_id, recover=False)

    def read_log(
        self, run_id: str, *, cell_id: str | None = None, tail_bytes: int = 64_000
    ) -> str:
        run = self.get(run_id)
        path = self._log_path(run, cell_id)
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - tail_bytes))
                return handle.read().decode(errors="replace")
        except OSError:
            return ""

    def read_log_chunk(
        self,
        run_id: str,
        *,
        cell_id: str | None = None,
        offset: int = 0,
        max_bytes: int = 64_000,
    ) -> tuple[str, int]:
        if offset < 0 or max_bytes < 1:
            raise ValueError("log offset must be non-negative and max_bytes positive")
        run = self.get(run_id)
        path = self._log_path(run, cell_id)
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(offset if offset <= size else 0)
                data = handle.read(max_bytes)
                return data.decode(errors="replace"), handle.tell()
        except OSError:
            return "", offset

    def follow_log(
        self,
        run_id: str,
        *,
        cell_id: str | None = None,
        poll_sec: float = 0.25,
    ) -> Iterator[str]:
        run = self.get(run_id)
        path = self._log_path(run, cell_id)
        position = 0
        while True:
            try:
                with path.open("r", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
            except OSError:
                chunk = ""
            if chunk:
                yield chunk
            current = self.get(run_id)
            if current.status not in {"starting", "running"}:
                if not chunk:
                    break
            time.sleep(poll_sec)

    def _recover(self, run_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        if metadata.get("status") not in {"starting", "running"}:
            return metadata
        pid = metadata.get("pid")
        if isinstance(pid, int) and _pid_alive(pid):
            return metadata
        message = "The managed process exited before recording a terminal state."
        run_dir = self._run_dir(run_id)
        mark_unfinished_cells(run_dir, "interrupted", message=message)
        write_run_manifest(
            self.repo_root,
            run_id,
            {
                "status": "interrupted",
                "ended_at": _now(),
                "error": message,
            },
        )
        return read_run_manifest(run_dir) or metadata

    def _managed(self, metadata: dict[str, Any]) -> ManagedRun:
        run_id = str(metadata.get("run_id") or "")
        run_dir = self._run_dir(run_id)
        configured_log = metadata.get("combined_log")
        log_path = Path(str(configured_log)) if configured_log else run_dir / "combined.log"
        return ManagedRun(
            run_id=run_id,
            status=str(metadata.get("status") or "unknown"),
            run_name=str(metadata.get("run_name") or run_id),
            experiment_id=str(metadata.get("experiment_id") or "unknown"),
            pid=metadata.get("pid") if isinstance(metadata.get("pid"), int) else None,
            created_at=metadata.get("created_at"),
            ended_at=metadata.get("ended_at"),
            run_dir=run_dir,
            log_path=log_path,
            metadata=metadata,
        )

    def _run_dir(self, run_id: str) -> Path:
        if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise ValueError(f"invalid run id: {run_id!r}")
        return self.runtime_root / run_id

    @staticmethod
    def _log_path(run: ManagedRun, cell_id: str | None) -> Path:
        return (
            run.run_dir / "logs" / f"{cell_id}.log" if cell_id else run.log_path
        )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for value in command:
        if hide_next:
            redacted.append("[redacted]")
            hide_next = False
            continue
        redacted.append(value)
        hide_next = value.lower() in {"--api-key", "--password", "--secret", "--token"}
    return redacted


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
