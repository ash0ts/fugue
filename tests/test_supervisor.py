from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fugue.bench.execution import write_run_manifest
from fugue.bench.supervisor import RunSupervisor


def test_detached_run_can_be_read_and_cancelled(tmp_path: Path) -> None:
    supervisor = RunSupervisor(tmp_path)
    run = supervisor.start_detached(
        run_id="run-detached",
        command=[
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(30)",
        ],
        env=os.environ.copy(),
        run_name="Detached",
        experiment_id="demo",
    )
    try:
        for _ in range(20):
            if "ready" in supervisor.read_log(run.run_id):
                break
            time.sleep(0.05)
        assert "ready" in supervisor.read_log(run.run_id)
        first, offset = supervisor.read_log_chunk(run.run_id)
        second, next_offset = supervisor.read_log_chunk(run.run_id, offset=offset)
        assert "ready" in first
        assert second == ""
        assert next_offset == offset
        cancelled = supervisor.cancel(run.run_id)
        assert cancelled.status == "cancelled"
    finally:
        current = supervisor.get(run.run_id, recover=False)
        if current.status in {"starting", "running"}:
            supervisor.cancel(run.run_id)


def test_orphaned_run_is_marked_interrupted(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        "run-orphan",
        {
            "status": "running",
            "pid": 99_999_999,
            "run_name": "Orphan",
            "experiment_id": "demo",
        },
    )

    run = RunSupervisor(tmp_path).get("run-orphan")

    assert run.status == "interrupted"
    assert "terminal state" in str(run.metadata["error"])
