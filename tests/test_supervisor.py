from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fugue.bench import supervisor as supervisor_module
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


def test_graceful_cancel_leaves_terminal_state_to_controller(tmp_path: Path) -> None:
    supervisor = RunSupervisor(tmp_path, cancel_grace_sec=2)
    run_path = tmp_path / ".fugue/runtime/run-graceful/run.json"
    script = """
import json
import os
import signal
import sys
import time

path = sys.argv[1]
def cancel(signum, frame):
    del signum, frame
    value = json.loads(open(path).read())
    value.update(status="cancelled", ended_at="controller", terminal_writer="controller")
    temp = path + ".controller.tmp"
    open(temp, "w").write(json.dumps(value))
    os.replace(temp, path)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, cancel)
print("ready", flush=True)
time.sleep(30)
"""
    run = supervisor.start_detached(
        run_id="run-graceful",
        command=[sys.executable, "-c", script, run_path.as_posix()],
        env=os.environ.copy(),
        run_name="Graceful",
        experiment_id="demo",
    )
    for _ in range(40):
        if "ready" in supervisor.read_log(run.run_id):
            break
        time.sleep(0.05)

    cancelled = supervisor.cancel(run.run_id)

    assert cancelled.status == "cancelled"
    assert cancelled.metadata["terminal_writer"] == "controller"
    assert cancelled.run_name == "Graceful"
    assert cancelled.experiment_id == "demo"
    assert cancelled.pid == run.pid
    assert "cancellation_forced" not in cancelled.metadata


def test_forced_cancel_records_open_prediction_as_truthfully_unclosed(
    tmp_path: Path,
) -> None:
    supervisor = RunSupervisor(tmp_path, cancel_grace_sec=0.05)
    run = supervisor.start_detached(
        run_id="run-forced",
        command=[
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(30)",
        ],
        env=os.environ.copy(),
        run_name="Forced",
        experiment_id="demo",
    )
    evaluations = run.run_dir / "evaluations.jsonl"
    evaluations.write_text(
        json.dumps(
            {
                "status": "prediction_open",
                "cell_id": "cell-a",
                "candidate_id": "candidate-a",
                "eval_predict_and_score_call_id": "call-a",
            }
        )
        + "\n"
    )
    write_run_manifest(
        tmp_path,
        run.run_id,
        {"evaluation_failures": ["preexisting observability failure"]},
    )
    for _ in range(40):
        if "ready" in supervisor.read_log(run.run_id):
            break
        time.sleep(0.05)

    cancelled = supervisor.cancel(run.run_id)

    assert cancelled.status == "cancelled"
    assert cancelled.metadata["cancellation_forced"] is True
    assert cancelled.metadata["observability_status"] == "failed"
    assert cancelled.metadata["evaluation_failures"] == [
        "preexisting observability failure",
        "cell-a: cancelled prediction was not closed",
    ]
    records = [json.loads(line) for line in evaluations.read_text().splitlines()]
    assert records[-1]["status"] == "cancelled_unclosed"
    assert records[-1]["eval_predict_and_score_call_id"] == "call-a"
    assert "cloud state is unknown" in records[-1]["error"]


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


def test_permission_error_does_not_mark_live_run_interrupted(
    tmp_path: Path, monkeypatch
) -> None:
    write_run_manifest(
        tmp_path,
        "run-restricted",
        {
            "status": "running",
            "pid": 123,
            "run_name": "Restricted",
            "experiment_id": "demo",
        },
    )

    def deny_signal(pid: int, signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr(supervisor_module.os, "kill", deny_signal)

    run = RunSupervisor(tmp_path).get("run-restricted")

    assert run.status == "running"


def test_cancel_cleanup_targets_only_snapshot_compose_projects(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / ".fugue/runtime/run-cleanup"
    run_dir.mkdir(parents=True)
    job_dir = tmp_path / "jobs/demo/job-a"
    (job_dir / "task-a__AbC123").mkdir(parents=True)
    outside = tmp_path / "outside/job-b"
    (outside / "task-b__escape").mkdir(parents=True)
    (run_dir / "input-lock.json").write_text(
        json.dumps(
            {
                "planned_matrix": [
                    {"result_path": "jobs/demo/job-a/result.json"},
                    {"result_path": outside.joinpath("result.json").as_posix()},
                ]
            }
        )
    )
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        commands.append(command)
        if command[1:3] == ["ps", "-aq"]:
            return subprocess.CompletedProcess(command, 0, "container-a\n", "")
        return subprocess.CompletedProcess(command, 0, "container-a\n", "")

    monkeypatch.setattr(supervisor_module.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(supervisor_module.subprocess, "run", run)

    projects, errors = supervisor_module._cleanup_run_compose_projects(
        tmp_path, run_dir
    )

    assert projects == ["task-a__abc123__env"]
    assert errors == []
    assert commands == [
        [
            "/docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.docker.compose.project=task-a__abc123__env",
        ],
        ["/docker", "rm", "-f", "container-a"],
    ]
