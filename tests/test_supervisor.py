from __future__ import annotations

import json
import os
import signal
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


def test_graceful_cancel_leaves_terminal_state_to_controller(
    tmp_path: Path, monkeypatch
) -> None:
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

    (run.run_dir / "cells.jsonl").write_text(
        json.dumps(
            {
                "cell_id": "active-cell",
                "status": "running",
                "harbor_process_group": 43_210,
            }
        )
        + "\n"
    )
    (run.run_dir / "input-lock.json").write_text(
        json.dumps({"planned_matrix": []})
    )
    signalled: list[tuple[int, signal.Signals]] = []
    original_getpgid = supervisor_module.os.getpgid

    def getpgid(pid: int) -> int:
        return pid if pid == 43_210 else original_getpgid(pid)

    def killpg(pid: int, signum: signal.Signals) -> None:
        signalled.append((pid, signum))

    monkeypatch.setattr(supervisor_module.os, "getpgid", getpgid)
    monkeypatch.setattr(supervisor_module.os, "killpg", killpg)
    cancelled = supervisor.cancel(run.run_id)

    assert cancelled.status == "cancelled"
    assert cancelled.metadata["terminal_writer"] == "controller"
    assert cancelled.run_name == "Graceful"
    assert cancelled.experiment_id == "demo"
    assert cancelled.pid == run.pid
    assert "cancellation_forced" not in cancelled.metadata
    assert signalled == [(43_210, signal.SIGTERM)]
    assert cancelled.metadata["cancellation_cleanup_status"] == "passed"
    receipt = json.loads(
        (run.run_dir / supervisor_module.CANCELLATION_CLEANUP_RECEIPT).read_text()
    )
    assert receipt["zero_orphans_verified"] is True
    assert receipt["graceful_termination_status"] == "passed"


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


def test_observational_read_does_not_recover_foreign_run(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        "run-foreign",
        {
            "status": "running",
            "pid": 99_999_999,
            "run_name": "Foreign worker",
            "experiment_id": "demo",
        },
    )

    run = RunSupervisor(tmp_path).get("run-foreign", recover=False)

    assert run.status == "running"
    assert "error" not in run.metadata


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
    job_dir = tmp_path / ".fugue/runtime/jobs/demo/run-cleanup/job-a"
    (job_dir / "task-a__AbC123").mkdir(parents=True)
    outside = tmp_path / "outside/job-b"
    (outside / "task-b__escape").mkdir(parents=True)
    (run_dir / "input-lock.json").write_text(
        json.dumps(
            {
                "planned_matrix": [
                    {
                        "result_path": (
                            ".fugue/runtime/jobs/demo/run-cleanup/job-a/result.json"
                        )
                    },
                    {"result_path": outside.joinpath("result.json").as_posix()},
                ]
            }
        )
    )
    commands: list[list[str]] = []
    container_queries = 0
    network_queries = 0

    def run(command: list[str], **kwargs):
        nonlocal container_queries, network_queries
        commands.append(command)
        if command[1:3] == ["ps", "-aq"]:
            container_queries += 1
            stdout = "container-a\n" if container_queries == 1 else ""
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if command[1:4] == ["network", "ls", "-q"]:
            network_queries += 1
            stdout = "network-a\n" if network_queries == 1 else ""
            return subprocess.CompletedProcess(command, 0, stdout, "")
        return subprocess.CompletedProcess(command, 0, "container-a\n", "")

    monkeypatch.setattr(supervisor_module.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(supervisor_module.subprocess, "run", run)

    result = supervisor_module._cleanup_run_compose_projects(tmp_path, run_dir)

    assert result.status == "passed"
    assert result.projects == ("task-a__abc123__env",)
    assert result.planned_jobs == 1
    assert result.started_jobs == 1
    assert result.never_started_jobs == 0
    assert result.removed_containers == ("container-a",)
    assert result.removed_networks == ("network-a",)
    assert result.remaining_containers == ()
    assert result.remaining_networks == ()
    assert result.zero_orphans_verified is True
    assert result.errors == ()
    assert commands == [
        [
            "/docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.docker.compose.project=task-a__abc123__env",
        ],
        ["/docker", "rm", "-f", "container-a"],
        [
            "/docker",
            "network",
            "ls",
            "-q",
            "--filter",
            "label=com.docker.compose.project=task-a__abc123__env",
        ],
        ["/docker", "network", "rm", "network-a"],
        [
            "/docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.docker.compose.project=task-a__abc123__env",
        ],
        [
            "/docker",
            "network",
            "ls",
            "-q",
            "--filter",
            "label=com.docker.compose.project=task-a__abc123__env",
        ],
    ]


def test_cancel_cleanup_removes_network_after_failed_compose_start(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / ".fugue/runtime/run-cleanup"
    run_dir.mkdir(parents=True)
    job_dir = tmp_path / "jobs/demo/job-a"
    (job_dir / "task-a__AbC123").mkdir(parents=True)
    (run_dir / "input-lock.json").write_text(
        json.dumps(
            {"planned_matrix": [{"result_path": "jobs/demo/job-a/result.json"}]}
        )
    )
    commands: list[list[str]] = []
    network_queries = 0

    def run(command: list[str], **kwargs):
        nonlocal network_queries
        commands.append(command)
        if command[1:3] == ["ps", "-aq"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1:4] == ["network", "ls", "-q"]:
            network_queries += 1
            stdout = "network-a\n" if network_queries == 1 else ""
            return subprocess.CompletedProcess(command, 0, stdout, "")
        return subprocess.CompletedProcess(command, 0, "network-a\n", "")

    monkeypatch.setattr(supervisor_module.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(supervisor_module.subprocess, "run", run)

    result = supervisor_module._cleanup_run_compose_projects(tmp_path, run_dir)

    assert result.status == "passed"
    assert result.projects == ("task-a__abc123__env",)
    assert result.errors == ()
    assert ["/docker", "rm", "-f", "container-a"] not in commands
    assert ["/docker", "network", "rm", "network-a"] in commands
    assert commands[-1] == [
        "/docker",
        "network",
        "ls",
        "-q",
        "--filter",
        "label=com.docker.compose.project=task-a__abc123__env",
    ]


def test_cancel_cleanup_distinguishes_never_started_jobs_and_writes_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "run-partial"
    run_dir = tmp_path / ".fugue/runtime" / run_id
    run_dir.mkdir(parents=True)
    started = tmp_path / ".fugue/runtime/jobs/demo/run-partial/started"
    (started / "task-started__Exact1").mkdir(parents=True)
    (run_dir / "input-lock.json").write_text(
        json.dumps(
            {
                "planned_matrix": [
                    {
                        "result_path": (
                            ".fugue/runtime/jobs/demo/run-partial/started/result.json"
                        )
                    },
                    {
                        "result_path": (
                            ".fugue/runtime/jobs/demo/run-partial/never/result.json"
                        )
                    },
                ]
            }
        )
    )
    write_run_manifest(
        tmp_path,
        run_id,
        {
            "status": "cancelled",
            "run_name": "Partial",
            "experiment_id": "demo",
        },
    )

    monkeypatch.setattr(supervisor_module.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )

    result = RunSupervisor(tmp_path).record_cancellation_cleanup(run_id)

    assert result.status == "passed"
    assert result.planned_jobs == 2
    assert result.started_jobs == 1
    assert result.never_started_jobs == 1
    assert result.zero_orphans_verified is True
    receipt_path = run_dir / supervisor_module.CANCELLATION_CLEANUP_RECEIPT
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "passed"
    assert receipt["planned_jobs"] == 2
    assert receipt["started_jobs"] == 1
    assert receipt["never_started_jobs"] == 1
    assert receipt["remaining_containers"] == []
    assert receipt["remaining_networks"] == []
    assert receipt["zero_orphans_verified"] is True
    manifest = supervisor_module.read_run_manifest(run_dir)
    assert manifest is not None
    assert manifest["cancellation_cleanup_status"] == "passed"
    assert manifest["cancellation_cleanup_receipt"]["sha256"] == receipt[
        "receipt_sha256"
    ]


def test_cancel_cleanup_fails_when_exact_project_container_remains(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / ".fugue/runtime/run-remains"
    run_dir.mkdir(parents=True)
    job_dir = tmp_path / ".fugue/runtime/jobs/demo/run-remains/job"
    (job_dir / "task__Exact1").mkdir(parents=True)
    (run_dir / "input-lock.json").write_text(
        json.dumps(
            {
                "planned_matrix": [
                    {
                        "result_path": (
                            ".fugue/runtime/jobs/demo/run-remains/job/result.json"
                        )
                    }
                ]
            }
        )
    )

    def run(command: list[str], **kwargs):
        if command[1:3] == ["ps", "-aq"]:
            return subprocess.CompletedProcess(command, 0, "container-a\n", "")
        if command[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(command, 1, "", "busy")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(supervisor_module.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(supervisor_module.subprocess, "run", run)

    result = supervisor_module._cleanup_run_compose_projects(tmp_path, run_dir)

    assert result.status == "failed"
    assert result.remaining_containers == ("container-a",)
    assert result.zero_orphans_verified is False
    assert result.errors == (
        "task__exact1__env: docker rm failed: busy",
    )
