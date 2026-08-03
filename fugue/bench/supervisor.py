from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock

from fugue.bench.execution import (
    latest_cell_records,
    list_run_manifests,
    mark_unfinished_cells,
    read_run_manifest,
    update_run_manifest,
    write_run_manifest,
)
from fugue.bench.files import atomic_write_json, latest_jsonl_records

CANCELLATION_CLEANUP_RECEIPT = "cancellation-cleanup.json"


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


@dataclass(frozen=True)
class CancellationCleanupResult:
    status: str
    projects: tuple[str, ...]
    planned_jobs: int
    started_jobs: int
    never_started_jobs: int
    removed_containers: tuple[str, ...] = ()
    removed_networks: tuple[str, ...] = ()
    remaining_containers: tuple[str, ...] = ()
    remaining_networks: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def zero_orphans_verified(self) -> bool:
        return (
            self.status == "passed"
            and not self.remaining_containers
            and not self.remaining_networks
        )

    def to_dict(self, *, run_id: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "status": self.status,
            "generated_at": _now(),
            "scope": {
                "kind": "exact_run_compose_projects",
                "compose_projects": list(self.projects),
                "selector": "com.docker.compose.project=<exact project>",
                "resource_types": ["container", "network"],
                "excluded": ["other Compose projects", "non-Docker resources"],
            },
            "planned_jobs": self.planned_jobs,
            "started_jobs": self.started_jobs,
            "never_started_jobs": self.never_started_jobs,
            "mutations_performed": bool(
                self.removed_containers or self.removed_networks
            ),
            "removed_containers": list(self.removed_containers),
            "removed_networks": list(self.removed_networks),
            "remaining_containers": list(self.remaining_containers),
            "remaining_networks": list(self.remaining_networks),
            "zero_orphans_verified": self.zero_orphans_verified,
            "errors": list(self.errors),
            "receipt_sha256": "",
        }


@dataclass(frozen=True)
class _ComposeProjectScope:
    projects: tuple[str, ...]
    planned_jobs: int
    started_jobs: int
    never_started_jobs: int
    errors: tuple[str, ...] = ()


class RunSupervisor:
    def __init__(self, repo_root: Path, *, cancel_grace_sec: float = 20.0):
        self.repo_root = repo_root.resolve()
        self.runtime_root = self.repo_root / ".fugue" / "runtime"
        self._processes: dict[int, subprocess.Popen[str]] = {}
        self.cancel_grace_sec = max(cancel_grace_sec, 0.0)

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
        threading.Thread(
            target=process.wait,
            name=f"fugue-reaper-{run_id}",
            daemon=True,
        ).start()
        write_run_manifest(
            self.repo_root,
            run_id,
            {
                "pid": process.pid,
                "process_group": process.pid,
            },
        )
        self._processes[process.pid] = process
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
            if run.status in {"cancelled", "interrupted"}:
                self.record_cancellation_cleanup(run_id)
                return self.get(run_id, recover=False)
            return run
        message = "Run cancelled by the operator."
        requested_at = _now()
        update_run_manifest(
            self.repo_root,
            run_id,
            lambda _: {
                "cancellation_requested_at": requested_at,
                "cancellation_reason": message,
            },
        )
        signal_errors = _signal_recorded_cell_groups(run.run_dir, signal.SIGTERM)
        if run.pid is not None:
            try:
                os.kill(run.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                signal_errors.append(
                    f"controller process {run.pid}: SIGTERM failed: {exc}"
                )

        deadline = time.monotonic() + self.cancel_grace_sec
        while time.monotonic() < deadline:
            current = read_run_manifest(run.run_dir) or {}
            if current.get("status") not in {"starting", "running"}:
                self.record_cancellation_cleanup(
                    run_id,
                    signal_errors=signal_errors,
                )
                return self.get(run_id, recover=False)
            if run.pid is None or not _pid_alive(run.pid):
                break
            time.sleep(0.05)

        current = read_run_manifest(run.run_dir) or {}
        if current.get("status") not in {"starting", "running"}:
            self.record_cancellation_cleanup(
                run_id,
                signal_errors=signal_errors,
            )
            return self.get(run_id, recover=False)
        if run.pid is not None and _pid_alive(run.pid):
            signal_errors.extend(
                _signal_recorded_cell_groups(run.run_dir, signal.SIGKILL)
            )
            process_group = run.metadata.get("process_group") or run.pid
            if isinstance(process_group, int):
                try:
                    os.killpg(process_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    signal_errors.append(
                        f"controller process group {process_group}: "
                        f"SIGKILL failed: {exc}"
                    )
            for _ in range(20):
                if not _pid_alive(run.pid):
                    break
                time.sleep(0.05)
            if _pid_alive(run.pid):
                raise RuntimeError(
                    f"run {run_id} did not exit after cancellation escalation"
                )

        failures = _record_forced_evaluation_cancellation(run.run_dir, message)
        mark_unfinished_cells(run.run_dir, "cancelled", message=message)

        def forced_cancellation(manifest: dict[str, Any]) -> dict[str, Any]:
            existing_failures = [
                str(item) for item in manifest.get("evaluation_failures") or ()
            ]
            combined_failures = list(dict.fromkeys([*existing_failures, *failures]))
            return {
                "status": "cancelled",
                "ended_at": _now(),
                "error": message,
                "cancellation_forced": True,
                "observability_status": (
                    "failed" if combined_failures else "cancelled"
                ),
                "evaluation_failures": combined_failures,
            }

        update_run_manifest(
            self.repo_root,
            run_id,
            forced_cancellation,
        )
        self.record_cancellation_cleanup(
            run_id,
            signal_errors=signal_errors,
        )
        return self.get(run_id, recover=False)

    def record_cancellation_cleanup(
        self,
        run_id: str,
        *,
        signal_errors: list[str] | tuple[str, ...] = (),
    ) -> CancellationCleanupResult:
        """Remove and verify only exact run-scoped Harbor resources."""

        run_dir = self._run_dir(run_id)
        receipt_path = run_dir / CANCELLATION_CLEANUP_RECEIPT
        with FileLock(f"{receipt_path}.lock"):
            result = _cleanup_run_compose_projects(self.repo_root, run_dir)
            payload = result.to_dict(run_id=run_id)
            prior_errors: list[str] = []
            prior_signal_errors: list[str] = []
            prior_removed_containers: list[str] = []
            prior_removed_networks: list[str] = []
            if receipt_path.is_file():
                try:
                    previous = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    previous = {}
                if isinstance(previous, dict):
                    prior_errors = [
                        str(value) for value in previous.get("errors") or ()
                    ]
                    prior_signal_errors = [
                        str(value)
                        for value in previous.get("signal_errors") or ()
                    ]
                    prior_removed_containers = [
                        str(value)
                        for value in previous.get("removed_containers") or ()
                    ]
                    prior_removed_networks = [
                        str(value)
                        for value in previous.get("removed_networks") or ()
                    ]
                    payload["previous_receipt_sha256"] = str(
                        previous.get("receipt_sha256") or ""
                    )
            payload["removed_containers"] = sorted(
                {
                    *prior_removed_containers,
                    *result.removed_containers,
                }
            )
            payload["removed_networks"] = sorted(
                {
                    *prior_removed_networks,
                    *result.removed_networks,
                }
            )
            payload["mutations_performed"] = bool(
                payload["removed_containers"] or payload["removed_networks"]
            )
            payload["prior_errors"] = sorted(set(prior_errors))
            payload["signal_errors"] = sorted(
                {*prior_signal_errors, *signal_errors}
            )
            payload["graceful_termination_status"] = (
                "failed" if payload["signal_errors"] else "passed"
            )
            payload["receipt_sha256"] = _receipt_digest(payload)
            atomic_write_json(receipt_path, payload)
        update_run_manifest(
            self.repo_root,
            run_id,
            lambda _: {
                "cancellation_cleanup_status": str(payload["status"]),
                "cancellation_cleanup_projects": list(result.projects),
                "cancellation_cleanup_errors": [
                    str(value) for value in payload["errors"]
                ],
                "cancellation_graceful_termination_status": str(
                    payload["graceful_termination_status"]
                ),
                "cancellation_signal_errors": [
                    str(value) for value in payload["signal_errors"]
                ],
                "cancellation_cleanup_receipt": {
                    "schema_version": 1,
                    "path": receipt_path.relative_to(self.repo_root).as_posix(),
                    "sha256": str(payload["receipt_sha256"]),
                    "status": str(payload["status"]),
                    "zero_orphans_verified": bool(
                        payload["zero_orphans_verified"]
                    ),
                },
            },
        )
        return result

    def read_log(
        self,
        run_id: str,
        *,
        cell_id: str | None = None,
        tail_bytes: int = 64_000,
        recover: bool = True,
    ) -> str:
        run = self.get(run_id, recover=recover)
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
        recover: bool = True,
    ) -> tuple[str, int]:
        if offset < 0 or max_bytes < 1:
            raise ValueError("log offset must be non-negative and max_bytes positive")
        run = self.get(run_id, recover=recover)
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
        recover: bool = True,
    ) -> Iterator[str]:
        run = self.get(run_id, recover=recover)
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
            current = self.get(run_id, recover=recover)
            if current.status not in {"starting", "running"}:
                if not chunk:
                    break
            time.sleep(poll_sec)

    def _recover(self, run_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        pid = metadata.get("pid")
        if isinstance(pid, int):
            self._reap_local_process(pid)
        if metadata.get("status") not in {"starting", "running"}:
            return metadata
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

    def _reap_local_process(self, pid: int, *, wait: bool = False) -> None:
        process = self._processes.get(pid)
        if process is None:
            return
        if wait:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return
        else:
            process.poll()
        if process.returncode is not None:
            self._processes.pop(pid, None)

    def _managed(self, metadata: dict[str, Any]) -> ManagedRun:
        run_id = str(metadata.get("run_id") or "")
        run_dir = self._run_dir(run_id)
        configured_log = metadata.get("combined_log")
        log_path = (
            Path(str(configured_log)) if configured_log else run_dir / "combined.log"
        )
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
        return run.run_dir / "logs" / f"{cell_id}.log" if cell_id else run.log_path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (ProcessLookupError, ValueError):
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


def _signal_recorded_cell_groups(
    run_dir: Path, signum: signal.Signals
) -> list[str]:
    errors: list[str] = []
    for record in latest_cell_records(run_dir / "cells.jsonl"):
        process_group = record.get("harbor_process_group")
        if not isinstance(process_group, int) or process_group <= 0:
            continue
        try:
            if os.getpgid(process_group) != process_group:
                continue
            os.killpg(process_group, signum)
        except ProcessLookupError:
            continue
        except OSError as exc:
            errors.append(
                f"Harbor process group {process_group}: {signum.name} failed: {exc}"
            )
    return errors


def _cleanup_run_compose_projects(
    repo_root: Path, run_dir: Path
) -> CancellationCleanupResult:
    scope = _compose_projects_from_snapshot(repo_root, run_dir)
    projects = list(scope.projects)
    if scope.errors:
        return CancellationCleanupResult(
            status="unavailable",
            projects=scope.projects,
            planned_jobs=scope.planned_jobs,
            started_jobs=scope.started_jobs,
            never_started_jobs=scope.never_started_jobs,
            errors=scope.errors,
        )
    if not projects:
        return CancellationCleanupResult(
            status="passed",
            projects=(),
            planned_jobs=scope.planned_jobs,
            started_jobs=scope.started_jobs,
            never_started_jobs=scope.never_started_jobs,
        )
    docker = shutil.which("docker")
    if docker is None:
        return CancellationCleanupResult(
            status="unavailable",
            projects=scope.projects,
            planned_jobs=scope.planned_jobs,
            started_jobs=scope.started_jobs,
            never_started_jobs=scope.never_started_jobs,
            errors=("docker is unavailable; Harbor resources were not removed",),
        )
    errors: list[str] = []
    removed_containers: list[str] = []
    removed_networks: list[str] = []
    remaining_containers: list[str] = []
    remaining_networks: list[str] = []
    for project in projects:
        container_ids, list_error = _docker_resource_ids(
            docker,
            resource="container",
            project=project,
        )
        if list_error is not None:
            errors.append(list_error)
            continue
        if container_ids:
            removed = subprocess.run(
                [docker, "rm", "-f", *container_ids],
                text=True,
                capture_output=True,
                check=False,
            )
            if removed.returncode != 0:
                errors.append(f"{project}: docker rm failed: {removed.stderr.strip()}")
            else:
                removed_containers.extend(container_ids)
        network_ids, network_error = _docker_resource_ids(
            docker,
            resource="network",
            project=project,
        )
        if network_error is not None:
            errors.append(network_error)
            continue
        if network_ids:
            removed = subprocess.run(
                [docker, "network", "rm", *network_ids],
                text=True,
                capture_output=True,
                check=False,
            )
            if removed.returncode != 0:
                errors.append(
                    f"{project}: docker network rm failed: "
                    f"{removed.stderr.strip()}"
                )
            else:
                removed_networks.extend(network_ids)

        verified_containers, container_verify_error = _docker_resource_ids(
            docker,
            resource="container",
            project=project,
        )
        if container_verify_error is not None:
            errors.append(container_verify_error)
        else:
            remaining_containers.extend(verified_containers)
        verified_networks, network_verify_error = _docker_resource_ids(
            docker,
            resource="network",
            project=project,
        )
        if network_verify_error is not None:
            errors.append(network_verify_error)
        else:
            remaining_networks.extend(verified_networks)
    status = (
        "failed"
        if remaining_containers or remaining_networks
        else "unavailable"
        if errors
        else "passed"
    )
    return CancellationCleanupResult(
        status=status,
        projects=scope.projects,
        planned_jobs=scope.planned_jobs,
        started_jobs=scope.started_jobs,
        never_started_jobs=scope.never_started_jobs,
        removed_containers=tuple(sorted(set(removed_containers))),
        removed_networks=tuple(sorted(set(removed_networks))),
        remaining_containers=tuple(sorted(set(remaining_containers))),
        remaining_networks=tuple(sorted(set(remaining_networks))),
        errors=tuple(errors),
    )


def _docker_resource_ids(
    docker: str,
    *,
    resource: str,
    project: str,
) -> tuple[list[str], str | None]:
    command = (
        [docker, "ps", "-aq"]
        if resource == "container"
        else [docker, "network", "ls", "-q"]
    )
    command.extend(
        ["--filter", f"label=com.docker.compose.project={project}"]
    )
    listed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if listed.returncode != 0:
        label = "docker ps" if resource == "container" else "docker network ls"
        return [], f"{project}: {label} failed: {listed.stderr.strip()}"
    return sorted({item for item in listed.stdout.splitlines() if item}), None


def _compose_projects_from_snapshot(
    repo_root: Path, run_dir: Path
) -> _ComposeProjectScope:
    try:
        snapshot = json.loads((run_dir / "input-lock.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _ComposeProjectScope(
            projects=(),
            planned_jobs=0,
            started_jobs=0,
            never_started_jobs=0,
            errors=("immutable run snapshot is unavailable",),
        )
    allowed_roots = (
        (repo_root / "jobs").resolve(),
        (repo_root / ".fugue" / "runtime" / "jobs").resolve(),
    )
    projects: set[str] = set()
    errors: list[str] = []
    planned_paths: set[Path] = set()
    for planned in snapshot.get("planned_matrix") or ():
        raw_result = planned.get("result_path") if isinstance(planned, dict) else None
        if not isinstance(raw_result, str) or not raw_result:
            continue
        result = Path(raw_result)
        job_dir = (
            result if result.is_absolute() else repo_root / result
        ).parent.resolve()
        if not any(job_dir.is_relative_to(root) for root in allowed_roots):
            continue
        planned_paths.add(job_dir)
    started_jobs = 0
    never_started_jobs = 0
    recorded_started = _recorded_started_job_directories(repo_root, run_dir)
    for job_dir in sorted(planned_paths):
        was_started = job_dir in recorded_started
        try:
            children = tuple(job_dir.iterdir())
        except FileNotFoundError:
            if was_started:
                started_jobs += 1
                errors.append(
                    f"started Harbor result directory is unavailable: {job_dir}"
                )
            else:
                never_started_jobs += 1
            continue
        except OSError as exc:
            errors.append(f"cannot inspect Harbor result directory {job_dir}: {exc}")
            continue
        job_projects: set[str] = set()
        for child in children:
            if not child.is_dir() or "__" not in child.name:
                continue
            project = f"{child.name.lower()}__env"
            if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*(?:__[a-z0-9_.-]+)+", project):
                job_projects.add(project)
        if job_projects:
            started_jobs += 1
            projects.update(job_projects)
        elif was_started:
            started_jobs += 1
            errors.append(
                f"started job has no exact Harbor Compose project: {job_dir}"
            )
        else:
            never_started_jobs += 1
    return _ComposeProjectScope(
        projects=tuple(sorted(projects)),
        planned_jobs=len(planned_paths),
        started_jobs=started_jobs,
        never_started_jobs=never_started_jobs,
        errors=tuple(errors),
    )


def _recorded_started_job_directories(
    repo_root: Path, run_dir: Path
) -> set[Path]:
    path = run_dir / "cells.jsonl"
    if not path.is_file():
        return set()
    directories: set[Path] = set()
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                process_group = record.get("harbor_process_group")
                raw_result = record.get("result_path")
                if (
                    not isinstance(process_group, int)
                    or process_group <= 0
                    or not isinstance(raw_result, str)
                    or not raw_result
                ):
                    continue
                result = Path(raw_result)
                directories.add(
                    (
                        result
                        if result.is_absolute()
                        else repo_root / result
                    ).parent.resolve()
                )
    except OSError:
        return set()
    return directories


def _receipt_digest(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["receipt_sha256"] = ""
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _record_forced_evaluation_cancellation(run_dir: Path, message: str) -> list[str]:
    path = run_dir / "evaluations.jsonl"
    if not path.is_file():
        return []
    latest = {
        str(record["cell_id"]): record
        for record in latest_jsonl_records(path, "cell_id")
    }
    failures: list[str] = []
    terminal = {"cancelled", "cancelled_unclosed", "failed", "finalized"}
    records: list[dict[str, Any]] = []
    for cell_id, record in latest.items():
        if record.get("status") in terminal:
            continue
        opened = record.get("status") == "prediction_open"
        status = "cancelled_unclosed" if opened else "cancelled"
        value = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "status": status,
            "recorded_at": _now(),
            "cell_id": cell_id,
            "candidate_id": record.get("candidate_id"),
            "eval_predict_and_score_call_id": record.get(
                "eval_predict_and_score_call_id"
            ),
            "error": (
                "Controller exited before the open evaluation prediction could be "
                "closed; cloud state is unknown."
                if opened
                else message
            ),
        }
        records.append(value)
        if opened:
            failures.append(f"{cell_id}: cancelled prediction was not closed")
    if records:
        with FileLock(f"{path}.lock"), path.open("a") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
    return failures


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
