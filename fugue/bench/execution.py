from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fugue.bench.job_config import RenderedJob

CellStatus = Literal["pending", "running", "passed", "failed", "not_applicable"]


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
    config_path: Path
    command: tuple[str, ...]
    env: dict[str, str]
    n_attempts: int
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
            "config_path": self.config_path.as_posix(),
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
                config_path=job.config_path,
                command=tuple(job.command),
                env=job.env,
                n_attempts=int(job.config.get("n_attempts") or 1),
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
    runner: Callable[..., Any] = subprocess.run,
) -> list[CellOutcome]:
    if max_workers < 1:
        raise ValueError("cell concurrency must be positive")
    run_ids = {cell.run_id for cell in cells}
    if len(run_ids) > 1:
        raise ValueError("all cells in one execution must share a run_id")
    cell_ids = [cell.id for cell in cells]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell ids must be unique within an execution")
    store = _CellStore(repo_root / ".fugue" / "runtime" / cells[0].run_id) if cells else None
    runnable: list[PlannedCell] = []
    outcomes: list[CellOutcome] = []
    for cell in cells:
        assert store is not None
        store.append(cell.record("pending"))
        if cell.applicable:
            runnable.append(cell)
        else:
            store.append(cell.record("not_applicable"))
            outcomes.append(CellOutcome(cell.id, "not_applicable"))

    def run_one(cell: PlannedCell) -> CellOutcome:
        assert store is not None
        store.append(cell.record("running"))
        started = datetime.now(UTC)
        try:
            result = runner(
                list(cell.command),
                check=False,
                env=cell.env,
                cwd=repo_root,
            )
            returncode = int(result.returncode)
            status: CellStatus = "passed" if returncode == 0 else "failed"
            outcome = CellOutcome(cell.id, status, returncode=returncode)
        except Exception as exc:
            outcome = CellOutcome(
                cell.id,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        ended = datetime.now(UTC)
        store.append(
            cell.record(
                outcome.status,
                returncode=outcome.returncode,
                error=outcome.error,
                started_at=started.isoformat(),
                ended_at=ended.isoformat(),
                wall_time_sec=(ended - started).total_seconds(),
            )
        )
        return outcome

    with ThreadPoolExecutor(max_workers=min(max_workers, len(runnable) or 1)) as pool:
        futures = [pool.submit(run_one, cell) for cell in runnable]
        for future in as_completed(futures):
            outcomes.append(future.result())
    return outcomes


def write_run_manifest(repo_root: Path, run_id: str, values: dict[str, Any]) -> Path:
    path = repo_root / ".fugue" / "runtime" / run_id / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
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


class _CellStore:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "cells.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        with self._lock, self.path.open("a") as handle:
            handle.write(line)
            handle.flush()
