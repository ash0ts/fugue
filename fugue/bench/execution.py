from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from filelock import FileLock

from fugue.redaction import redact_text, secrets_from_env

if TYPE_CHECKING:
    from fugue.bench.job_config import RenderedJob

CellStatus = Literal[
    "pending",
    "running",
    "passed",
    "failed",
    "not_applicable",
    "cancelled",
    "interrupted",
]
RunStatus = Literal[
    "starting",
    "running",
    "passed",
    "failed",
    "cancelled",
    "interrupted",
]
EventCallback = Callable[[dict[str, Any]], None]
CellStartedCallback = Callable[["PlannedCell"], Mapping[str, str] | None]
CellFinishedCallback = Callable[["PlannedCell", "CellOutcome"], None]


@dataclass(frozen=True)
class PlannedCell:
    id: str
    run_id: str
    run_name: str
    workload_id: str
    task_id: str
    harness: str
    context_system_id: str
    variant_id: str
    model_provider: str
    model: str
    trial_index: int
    comparison_example_id: str
    candidate_id: str
    config_path: Path
    result_path: Path
    command: tuple[str, ...]
    env: dict[str, str]
    n_attempts: int
    evaluation_case: dict[str, Any] | None = None
    evaluation_rubrics: tuple[dict[str, Any], ...] = ()
    scorer_hashes: dict[str, str] | None = None
    scorer_refs: tuple[str, ...] = ()
    applicable: bool = True
    skip_reason: str | None = None

    def record(self, status: CellStatus, **values: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "cell_id": self.id,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "workload_id": self.workload_id,
            "task_id": self.task_id,
            "harness": self.harness,
            "context_system_id": self.context_system_id,
            "variant_id": self.variant_id,
            "model_provider": self.model_provider,
            "model": self.model,
            "trial_index": self.trial_index,
            "comparison_example_id": self.comparison_example_id,
            "candidate_id": self.candidate_id,
            "config_path": self.config_path.as_posix(),
            "result_path": self.result_path.as_posix(),
            "command": list(self.command),
            "n_attempts": self.n_attempts,
            "status": status,
            "skip_reason": self.skip_reason,
            "recorded_at": datetime.now(UTC).isoformat(),
            **values,
        }


@dataclass(frozen=True)
class CellOutcome:
    cell_id: str
    status: CellStatus
    returncode: int | None = None
    error: str | None = None


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:10]}"


def plan_cells(
    jobs: list[RenderedJob], *, run_id: str, run_name: str
) -> list[PlannedCell]:
    cells: list[PlannedCell] = []
    for job in jobs:
        identity = ":".join(
            (
                run_id,
                job.workload_id,
                job.task_id,
                job.harness,
                job.context_system_id,
                job.variant_id,
                str(job.trial_index),
            )
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
        cells.append(
            PlannedCell(
                id=f"cell-{digest}",
                run_id=run_id,
                run_name=run_name,
                workload_id=job.workload_id,
                task_id=job.task_id,
                harness=job.harness,
                context_system_id=job.context_system_id,
                variant_id=job.variant_id,
                model_provider=job.route.provider,
                model=job.route.display_model,
                trial_index=job.trial_index,
                comparison_example_id=job.comparison_example_id,
                candidate_id=job.candidate_id,
                config_path=job.config_path,
                result_path=Path(str(job.config["jobs_dir"]))
                / job.job_name
                / "result.json",
                command=tuple(job.command),
                env=job.env,
                n_attempts=int(job.config.get("n_attempts") or 1),
                evaluation_case=job.evaluation_case,
                evaluation_rubrics=job.evaluation_rubrics,
                scorer_hashes=job.scorer_hashes,
                scorer_refs=job.scorer_refs,
                applicable=job.applicable,
                skip_reason=job.skip_reason,
            )
        )
    return cells


def execute_cells(
    cells: list[PlannedCell],
    *,
    repo_root: Path,
    max_workers: int,
    runner: Callable[..., Any] | None = None,
    event_callback: EventCallback | None = None,
    cell_started: CellStartedCallback | None = None,
    cell_finished: CellFinishedCallback | None = None,
) -> list[CellOutcome]:
    if max_workers < 1:
        raise ValueError("cell concurrency must be positive")
    run_ids = {cell.run_id for cell in cells}
    if len(run_ids) > 1:
        raise ValueError("all cells in one execution must share a run_id")
    cell_ids = [cell.id for cell in cells]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell ids must be unique within an execution")
    store = (
        _RunStore(repo_root / ".fugue" / "runtime" / cells[0].run_id, event_callback)
        if cells
        else None
    )
    runnable: list[PlannedCell] = []
    outcomes: list[CellOutcome] = []
    for cell in cells:
        assert store is not None
        store.append_cell(cell.record("pending"))
        store.append_event("cell_state", cell=cell, status="pending")
        if cell.applicable:
            runnable.append(cell)
        else:
            store.append_cell(cell.record("not_applicable"))
            store.append_event(
                "cell_state",
                cell=cell,
                status="not_applicable",
                message=cell.skip_reason,
            )
            outcomes.append(CellOutcome(cell.id, "not_applicable"))

    def run_one(cell: PlannedCell) -> CellOutcome:
        assert store is not None
        store.append_cell(cell.record("running"))
        store.append_event("cell_state", cell=cell, status="running")
        started = datetime.now(UTC)
        execution_env = dict(cell.env)
        if cell_started is not None:
            try:
                execution_env.update(cell_started(cell) or {})
            except Exception as exc:
                store.append_event(
                    "observability_error",
                    cell=cell,
                    message=f"{type(exc).__name__}: {exc}",
                )
        try:
            if runner is None:
                returncode = _run_cell_process(cell, repo_root, store, execution_env)
            else:
                result = runner(
                    list(cell.command),
                    check=False,
                    env=execution_env,
                    cwd=repo_root,
                )
                returncode = int(result.returncode)
            trial_error = (
                _harbor_job_error(cell, repo_root)
                if runner is None and returncode == 0
                else None
            )
            status: CellStatus = (
                "passed" if returncode == 0 and trial_error is None else "failed"
            )
            outcome = CellOutcome(
                cell.id,
                status,
                returncode=returncode,
                error=trial_error,
            )
        except Exception as exc:
            outcome = CellOutcome(
                cell.id,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        ended = datetime.now(UTC)
        if cell_finished is not None:
            try:
                cell_finished(cell, outcome)
            except Exception as exc:
                store.append_event(
                    "observability_error",
                    cell=cell,
                    message=f"{type(exc).__name__}: {exc}",
                )
        store.append_cell(
            cell.record(
                outcome.status,
                returncode=outcome.returncode,
                error=outcome.error,
                started_at=started.isoformat(),
                ended_at=ended.isoformat(),
                wall_time_sec=(ended - started).total_seconds(),
            )
        )
        store.append_event(
            "cell_state",
            cell=cell,
            status=outcome.status,
            returncode=outcome.returncode,
            message=outcome.error,
            wall_time_sec=(ended - started).total_seconds(),
        )
        return outcome

    with ThreadPoolExecutor(max_workers=min(max_workers, len(runnable) or 1)) as pool:
        futures = [pool.submit(run_one, cell) for cell in runnable]
        for future in as_completed(futures):
            outcomes.append(future.result())
    return outcomes


def _run_cell_process(
    cell: PlannedCell,
    repo_root: Path,
    store: _RunStore,
    env: Mapping[str, str],
) -> int:
    log_path = store.logs_dir / f"{cell.id}.log"
    process = subprocess.Popen(
        list(cell.command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=dict(env),
        cwd=repo_root,
    )
    secrets = secrets_from_env(env)
    with log_path.open("a") as log:
        assert process.stdout is not None
        for line in process.stdout:
            safe_line = redact_text(line, secrets)
            log.write(safe_line)
            log.flush()
            print(safe_line, end="", flush=True)
            store.append_event("log", cell=cell, chunk=safe_line)
    return process.wait()


def _harbor_job_error(cell: PlannedCell, repo_root: Path) -> str | None:
    path = cell.result_path
    if not path.is_absolute():
        path = repo_root / path
    try:
        result = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        return f"Harbor did not produce a readable job result: {exc}"
    stats = result.get("stats") or {}
    errored = int(stats.get("n_errored_trials") or 0)
    cancelled = int(stats.get("n_cancelled_trials") or 0)
    if errored:
        return f"{errored} Harbor trial(s) errored"
    if cancelled:
        return f"{cancelled} Harbor trial(s) were cancelled"
    return None


def write_run_manifest(repo_root: Path, run_id: str, values: dict[str, Any]) -> Path:
    path = repo_root / ".fugue" / "runtime" / run_id / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(".lock").as_posix()):
        existing = read_run_manifest(path.parent) or {}
        created_at = existing.get("created_at") or datetime.now(UTC).isoformat()
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(
            json.dumps(
                {
                    **existing,
                    "schema_version": 2,
                    "run_id": run_id,
                    "created_at": created_at,
                    "updated_at": datetime.now(UTC).isoformat(),
                    **values,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        os.replace(temp, path)
    return path


def read_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir if run_dir.name == "run.json" else run_dir / "run.json"
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def list_run_manifests(repo_root: Path) -> list[dict[str, Any]]:
    runtime = repo_root / ".fugue" / "runtime"
    if not runtime.exists():
        return []
    values = [
        value
        for path in runtime.glob("*/run.json")
        if (value := read_run_manifest(path)) is not None
    ]
    return sorted(
        values,
        key=lambda item: str(item.get("created_at") or item.get("run_id") or ""),
        reverse=True,
    )


def mark_unfinished_cells(
    run_dir: Path,
    status: Literal["cancelled", "interrupted"],
    *,
    message: str,
) -> None:
    state_path = run_dir / "cells.jsonl"
    latest = latest_cell_records(state_path)
    store = _RunStore(run_dir)
    for record in latest:
        if record.get("status") not in {"pending", "running"}:
            continue
        updated = {
            **record,
            "status": status,
            "error": message,
            "recorded_at": datetime.now(UTC).isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
        }
        store.append_cell(updated)
        store.append_event(
            "cell_state",
            cell_id=str(record.get("cell_id") or ""),
            status=status,
            message=message,
        )


def latest_cell_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("cell_id"):
            latest[str(record["cell_id"])] = record
    return list(latest.values())


class _RunStore:
    def __init__(
        self, run_dir: Path, event_callback: EventCallback | None = None
    ) -> None:
        self.run_dir = run_dir
        self.cells_path = run_dir / "cells.jsonl"
        self.events_path = run_dir / "events.jsonl"
        self.logs_dir = run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._event_callback = event_callback
        self._lock = threading.Lock()
        self._cells_file_lock = FileLock(f"{self.cells_path}.lock")
        self._events_file_lock = FileLock(f"{self.events_path}.lock")

    def append_cell(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        with self._lock, self._cells_file_lock, self.cells_path.open("a") as handle:
            handle.write(line)
            handle.flush()

    def append_event(
        self,
        event: str,
        *,
        cell: PlannedCell | None = None,
        cell_id: str | None = None,
        **data: Any,
    ) -> None:
        record = {
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "event": event,
            "recorded_at": datetime.now(UTC).isoformat(),
            "run_id": cell.run_id if cell else self.run_dir.name,
            "cell_id": cell.id if cell else cell_id,
            **data,
        }
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        with self._lock, self._events_file_lock, self.events_path.open("a") as handle:
            handle.write(line)
            handle.flush()
        if self._event_callback is not None:
            self._event_callback(record)
