from __future__ import annotations

import hashlib
import json
import math
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from filelock import FileLock

from fugue.bench.candidates import attempt_id, attempt_identity, stable_digest
from fugue.bench.executables import resolve_console_script
from fugue.bench.files import atomic_write_json, latest_jsonl_records
from fugue.bench.files import terminate_process_group as _terminate_process_group
from fugue.bench.harbor_outcome import (
    HARBOR_TERMINAL_CLASSIFIER_DIGEST,
    classify_harbor_terminal,
)
from fugue.bench.sandbox_policy import verify_harbor_job_attestation
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
EventCallback = Callable[[dict[str, Any]], None]
ExecutionKind = Literal["agent", "provider_diagnostic"]
BenchmarkOutcome = Literal["passed", "failed", "unscored", "not_applicable"]
RuntimeOutcome = Literal[
    "completed",
    "timed_out",
    "cancelled",
    "interrupted",
    "not_started",
    "not_applicable",
]
CellTerminalKind = Literal[
    "success",
    "task_failure",
    "agent_timeout",
    "cancelled",
    "runner_start_failure",
    "sandbox_lost",
    "transport_interrupted",
    "routing_failure",
    "identity_drift",
    "privacy_failure",
    "evidence_failure",
    "cleanup_failure",
    "execution_failure",
]
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
    execution_fingerprint: str
    config_path: Path
    result_path: Path
    command: tuple[str, ...]
    env: dict[str, str]
    n_attempts: int
    execution_kind: ExecutionKind = "agent"
    outer_wall_time_sec: int | None = None
    execution_limits_digest: str = ""
    context_delivery: str = "portable"
    expected_evidence_paths: tuple[str, ...] = ()
    evaluation_asset_lock_sha256: str = ""
    run_snapshot_sha256: str = ""
    source_commit: str = ""
    source_tree: str = ""
    source_dirty_digest: str = ""
    evaluation_case: dict[str, Any] | None = None
    evaluation_rubrics: tuple[dict[str, Any], ...] = ()
    scorer_hashes: dict[str, str] | None = None
    scorer_refs: tuple[str, ...] = ()
    applicable: bool = True
    skip_reason: str | None = None
    config_sha256: str = ""
    runtime_assets: tuple[tuple[str, str], ...] = ()
    approved_comparison: dict[str, Any] = field(default_factory=dict)
    integration_provenance: tuple[dict[str, Any], ...] = ()
    physical_execution_id: str = ""
    retry_ordinal: int = 0

    @property
    def attempt_identity(self) -> dict[str, Any]:
        return attempt_identity(
            task_id=self.task_id,
            arm=self.variant_id,
            harness=self.harness,
            attempt=self.trial_index,
            candidate=self.candidate_id,
            runtime=self.execution_fingerprint,
        )

    @property
    def attempt_id(self) -> str:
        return attempt_id(**self.attempt_identity)

    def record(self, status: CellStatus, **values: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "cell_id": self.id,
            "attempt_id": self.attempt_id,
            "attempt_identity": self.attempt_identity,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "workload_id": self.workload_id,
            "task_id": self.task_id,
            "harness": self.harness,
            "context_system_id": self.context_system_id,
            "context_delivery": self.context_delivery,
            "variant_id": self.variant_id,
            "model_provider": self.model_provider,
            "model": self.model,
            "trial_index": self.trial_index,
            "comparison_example_id": self.comparison_example_id,
            "candidate_id": self.candidate_id,
            "execution_fingerprint": self.execution_fingerprint,
            "execution_kind": self.execution_kind,
            "physical_execution_id": self.physical_execution_id or None,
            "retry_ordinal": self.retry_ordinal,
            "outer_wall_time_sec": self.outer_wall_time_sec,
            "execution_limits_digest": self.execution_limits_digest or None,
            "applicable": self.applicable,
            "config_path": self.config_path.as_posix(),
            "result_path": self.result_path.as_posix(),
            "command": list(self.command),
            "n_attempts": self.n_attempts,
            "status": status,
            "skip_reason": self.skip_reason,
            "config_sha256": self.config_sha256,
            "runtime_assets": [list(item) for item in self.runtime_assets],
            "run_snapshot_sha256": self.run_snapshot_sha256 or None,
            "source_commit": self.source_commit or None,
            "source_tree": self.source_tree or None,
            "source_dirty_digest": self.source_dirty_digest or None,
            "integration_provenance": list(self.integration_provenance),
            **(
                {"approved_comparison": self.approved_comparison}
                if self.approved_comparison
                else {}
            ),
            "recorded_at": datetime.now(UTC).isoformat(),
            **values,
        }


@dataclass(frozen=True)
class CellOutcome:
    cell_id: str
    status: CellStatus
    returncode: int | None = None
    error: str | None = None
    benchmark_outcome: BenchmarkOutcome = "unscored"
    reward: float | None = None
    runtime_outcome: RuntimeOutcome = "not_started"
    terminal_kind: CellTerminalKind | None = None


class TypedInfrastructureFailure(RuntimeError):
    """A runner-provided, predeclared infrastructure interruption.

    Generic exceptions are deliberately excluded: retry permission must come
    from structured runner evidence, never from exception text.
    """

    terminal_kind: Literal[
        "runner_start_failure", "sandbox_lost", "transport_interrupted"
    ]

    def __init__(
        self,
        terminal_kind: Literal[
            "runner_start_failure", "sandbox_lost", "transport_interrupted"
        ],
        message: str,
    ) -> None:
        if terminal_kind not in {
            "runner_start_failure",
            "sandbox_lost",
            "transport_interrupted",
        }:
            raise ValueError("unknown infrastructure terminal kind")
        super().__init__(message)
        self.terminal_kind = terminal_kind


@dataclass(frozen=True)
class _HarborJobResult:
    error: str | None
    benchmark_outcome: BenchmarkOutcome
    reward: float | None = None
    terminal_kind: CellTerminalKind | None = None
    runtime_outcome: RuntimeOutcome | None = None


class _CellWallTimeExceeded(RuntimeError):
    pass


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:10]}"


def plan_cells(
    jobs: list[RenderedJob],
    *,
    run_id: str,
    run_name: str,
    scheduling_seed: str | None = None,
    verify_inputs: bool = True,
    approved_comparison: Mapping[str, Any] | None = None,
) -> list[PlannedCell]:
    cells: list[PlannedCell] = []
    for job in jobs:
        stable_attempt_id = attempt_id(
            task_id=job.task_id,
            arm=job.variant_id,
            harness=job.harness,
            attempt=job.trial_index,
            candidate=job.candidate_id,
            runtime=job.resolved_candidate.execution_fingerprint,
        )
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
                execution_fingerprint=job.resolved_candidate.execution_fingerprint,
                execution_kind=job.execution_kind,
                config_path=job.config_path,
                result_path=job.result_path,
                command=tuple(job.command),
                env={**job.env, "FUGUE_ATTEMPT_ID": stable_attempt_id},
                n_attempts=job.n_attempts,
                outer_wall_time_sec=job.outer_wall_time_sec,
                execution_limits_digest=job.execution_limits_digest,
                context_delivery=job.context_delivery,
                expected_evidence_paths=job.expected_evidence_paths,
                evaluation_case=job.evaluation_case,
                evaluation_rubrics=job.evaluation_rubrics,
                scorer_hashes=job.scorer_hashes,
                scorer_refs=job.scorer_refs,
                applicable=job.applicable,
                skip_reason=job.skip_reason,
                config_sha256=(_path_digest(job.config_path) if verify_inputs else ""),
                runtime_assets=(
                    tuple(
                        (path.as_posix(), _path_digest(path))
                        for path in job.generated_runtime_files
                    )
                    if verify_inputs
                    else ()
                ),
                approved_comparison=dict(approved_comparison or {}),
                integration_provenance=job.integration_provenance,
            )
        )
    return schedule_cells(cells, scheduling_seed)


def schedule_cells(
    cells: list[PlannedCell], scheduling_seed: str | None
) -> list[PlannedCell]:
    if not scheduling_seed:
        return cells
    return sorted(
        cells,
        key=lambda cell: hashlib.sha256(
            ":".join(
                (
                    scheduling_seed,
                    cell.workload_id,
                    cell.task_id,
                    cell.harness,
                    cell.context_system_id,
                    cell.variant_id,
                    str(cell.trial_index),
                )
            ).encode()
        ).hexdigest(),
    )


def execute_cells(  # noqa: C901 - owns the complete cell lifecycle
    cells: list[PlannedCell],
    *,
    repo_root: Path,
    max_workers: int,
    runner: Callable[..., Any] | None = None,
    event_callback: EventCallback | None = None,
    cell_started: CellStartedCallback | None = None,
    require_cell_started_success: bool = False,
    cell_finished: CellFinishedCallback | None = None,
    cancellation_event: threading.Event | None = None,
    cancellation_message: str = "Run cancelled by the operator.",
    redaction_secrets: Sequence[str] = (),
) -> list[CellOutcome]:
    if max_workers < 1:
        raise ValueError("cell concurrency must be positive")
    run_ids = {cell.run_id for cell in cells}
    if len(run_ids) > 1:
        raise ValueError("all cells in one execution must share a run_id")
    cell_ids = [cell.id for cell in cells]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell ids must be unique within an execution")
    secrets = tuple(
        sorted(
            set(redaction_secrets)
            | {value for cell in cells for value in secrets_from_env(cell.env)},
            key=len,
            reverse=True,
        )
    )

    def safe_outcome(outcome: CellOutcome) -> CellOutcome:
        return CellOutcome(
            cell_id=outcome.cell_id,
            status=outcome.status,
            returncode=outcome.returncode,
            error=(redact_text(outcome.error, secrets) if outcome.error else None),
            benchmark_outcome=outcome.benchmark_outcome,
            reward=outcome.reward,
            runtime_outcome=outcome.runtime_outcome,
            terminal_kind=outcome.terminal_kind,
        )

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
            store.append_cell(
                cell.record(
                    "not_applicable",
                    benchmark_outcome="not_applicable",
                    runtime_outcome="not_applicable",
                )
            )
            store.append_event(
                "cell_state",
                cell=cell,
                status="not_applicable",
                message=cell.skip_reason,
            )
            outcomes.append(
                CellOutcome(
                    cell.id,
                    "not_applicable",
                    benchmark_outcome="not_applicable",
                    runtime_outcome="not_applicable",
                )
            )

    def run_one(cell: PlannedCell) -> CellOutcome:
        assert store is not None
        if cancellation_event is not None and cancellation_event.is_set():
            outcome = safe_outcome(
                CellOutcome(
                    cell.id,
                    "cancelled",
                    error=cancellation_message,
                    runtime_outcome="cancelled",
                    terminal_kind="cancelled",
                )
            )
            ended = datetime.now(UTC)
            store.append_cell(
                cell.record(
                    "cancelled",
                    error=outcome.error,
                    benchmark_outcome=outcome.benchmark_outcome,
                    runtime_outcome="cancelled",
                    terminal_kind=outcome.terminal_kind,
                    ended_at=ended.isoformat(),
                )
            )
            store.append_event(
                "cell_state",
                cell=cell,
                status="cancelled",
                message=outcome.error,
                terminal_kind=outcome.terminal_kind,
            )
            return outcome
        try:
            _verify_cell_inputs(cell, repo_root)
        except Exception as exc:
            error = redact_text(
                f"immutable run input verification failed: {exc}", secrets
            )
            outcome = CellOutcome(
                cell.id,
                "failed",
                error=error,
                terminal_kind="identity_drift",
            )
            store.append_cell(
                cell.record(
                    "failed",
                    error=error,
                    benchmark_outcome="unscored",
                    runtime_outcome="not_started",
                    terminal_kind=outcome.terminal_kind,
                    ended_at=datetime.now(UTC).isoformat(),
                )
            )
            store.append_event(
                "cell_state",
                cell=cell,
                status="failed",
                message=error,
                terminal_kind=outcome.terminal_kind,
            )
            return outcome
        store.append_cell(cell.record("running"))
        store.append_event("cell_state", cell=cell, status="running")
        started = datetime.now(UTC)
        execution_env, cell_started_called, start_failure = _initialize_cell_evidence(
            cell,
            store=store,
            started=started,
            cell_started=cell_started,
            require_success=require_cell_started_success,
            cancellation_event=cancellation_event,
            redaction_secrets=secrets,
        )
        if start_failure is not None:
            return start_failure
        if cancellation_event is not None and cancellation_event.is_set():
            outcome = safe_outcome(
                CellOutcome(
                    cell.id,
                    "cancelled",
                    error=cancellation_message,
                    runtime_outcome="cancelled",
                    terminal_kind="cancelled",
                )
            )
            ended = datetime.now(UTC)
            if cell_started_called and cell_finished is not None:
                try:
                    cell_finished(cell, outcome)
                except Exception as exc:
                    store.append_event(
                        "observability_error",
                        cell=cell,
                        message=redact_text(f"{type(exc).__name__}: {exc}", secrets),
                    )
            store.append_cell(
                cell.record(
                    "cancelled",
                    error=outcome.error,
                    benchmark_outcome=outcome.benchmark_outcome,
                    runtime_outcome="cancelled",
                    terminal_kind=outcome.terminal_kind,
                    started_at=started.isoformat(),
                    ended_at=ended.isoformat(),
                    wall_time_sec=(ended - started).total_seconds(),
                )
            )
            store.append_event(
                "cell_state",
                cell=cell,
                status="cancelled",
                message=outcome.error,
                terminal_kind=outcome.terminal_kind,
                wall_time_sec=(ended - started).total_seconds(),
            )
            return outcome
        try:
            if runner is None:
                returncode = _run_cell_process(
                    cell,
                    repo_root,
                    store,
                    execution_env,
                    **(
                        {"cancellation_event": cancellation_event}
                        if cancellation_event is not None
                        else {}
                    ),
                )
            else:
                result = runner(
                    list(cell.command),
                    check=False,
                    env=execution_env,
                    cwd=repo_root,
                )
                returncode = int(result.returncode)
            harbor_result = (
                _harbor_job_result(cell, repo_root)
                if runner is None and returncode == 0 and cell.execution_kind == "agent"
                else _HarborJobResult(None, "unscored")
            )
            trial_error = harbor_result.error
            cancellation_requested = bool(
                cancellation_event is not None and cancellation_event.is_set()
            )
            if harbor_result.terminal_kind is not None:
                terminal_kind = harbor_result.terminal_kind
                status = (
                    "passed"
                    if terminal_kind == "success"
                    else "cancelled"
                    if terminal_kind == "cancelled"
                    else "failed"
                )
                runtime_outcome = harbor_result.runtime_outcome or (
                    "completed" if status != "cancelled" else "cancelled"
                )
            elif cancellation_requested and (
                returncode != 0 or trial_error is not None
            ):
                status = "cancelled"
                trial_error = cancellation_message
                terminal_kind = "cancelled"
                runtime_outcome = "cancelled"
            else:
                status = (
                    "passed" if returncode == 0 and trial_error is None else "failed"
                )
                terminal_kind = "success" if status == "passed" else "task_failure"
                runtime_outcome = "completed"
            outcome = CellOutcome(
                cell.id,
                status,
                returncode=returncode,
                error=trial_error,
                benchmark_outcome=harbor_result.benchmark_outcome,
                reward=harbor_result.reward,
                runtime_outcome=runtime_outcome,
                terminal_kind=terminal_kind,
            )
        except TypedInfrastructureFailure as exc:
            outcome = CellOutcome(
                cell.id,
                "failed",
                error=str(exc),
                runtime_outcome="not_started",
                terminal_kind=exc.terminal_kind,
            )
        except _CellWallTimeExceeded as exc:
            outcome = CellOutcome(
                cell.id,
                "failed",
                error=str(exc),
                benchmark_outcome="unscored",
                runtime_outcome="timed_out",
                terminal_kind="agent_timeout",
            )
        except Exception as exc:
            outcome = CellOutcome(
                cell.id,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                runtime_outcome="not_started",
                terminal_kind="execution_failure",
            )
        outcome = safe_outcome(outcome)
        if cell.physical_execution_id:
            try:
                _write_physical_runner_terminal_observation(
                    repo_root=repo_root,
                    cell=cell,
                    outcome=outcome,
                )
            except Exception as exc:
                outcome = CellOutcome(
                    cell.id,
                    "failed",
                    error=(
                        "required physical runner terminal receipt failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    benchmark_outcome="unscored",
                    runtime_outcome="not_started",
                    terminal_kind="evidence_failure",
                )
        ended = datetime.now(UTC)
        if cell_finished is not None:
            try:
                cell_finished(cell, outcome)
            except Exception as exc:
                store.append_event(
                    "observability_error",
                    cell=cell,
                    message=redact_text(f"{type(exc).__name__}: {exc}", secrets),
                )
        store.append_cell(
            cell.record(
                outcome.status,
                returncode=outcome.returncode,
                error=outcome.error,
                benchmark_outcome=outcome.benchmark_outcome,
                reward=outcome.reward,
                runtime_outcome=outcome.runtime_outcome,
                terminal_kind=outcome.terminal_kind,
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
            benchmark_outcome=outcome.benchmark_outcome,
            reward=outcome.reward,
            runtime_outcome=outcome.runtime_outcome,
            terminal_kind=outcome.terminal_kind,
            wall_time_sec=(ended - started).total_seconds(),
        )
        return outcome

    with ThreadPoolExecutor(max_workers=min(max_workers, len(runnable) or 1)) as pool:
        futures = [pool.submit(run_one, cell) for cell in runnable]
        for future in as_completed(futures):
            outcomes.append(future.result())
    return outcomes


def _initialize_cell_evidence(
    cell: PlannedCell,
    *,
    store: _RunStore,
    started: datetime,
    cell_started: CellStartedCallback | None,
    require_success: bool,
    cancellation_event: threading.Event | None,
    redaction_secrets: Sequence[str],
) -> tuple[dict[str, str], bool, CellOutcome | None]:
    execution_env = dict(cell.env)
    if cell_started is None or (
        cancellation_event is not None and cancellation_event.is_set()
    ):
        return execution_env, False, None
    try:
        execution_env.update(cell_started(cell) or {})
        return execution_env, True, None
    except Exception as exc:
        error = redact_text(f"{type(exc).__name__}: {exc}", redaction_secrets)
        store.append_event("observability_error", cell=cell, message=error)
        if not require_success:
            return execution_env, True, None
        outcome = CellOutcome(
            cell.id,
            "failed",
            error=f"required live-evidence initialization failed: {error}",
            runtime_outcome="not_started",
            terminal_kind="evidence_failure",
        )
        ended = datetime.now(UTC)
        wall_time = (ended - started).total_seconds()
        store.append_cell(
            cell.record(
                "failed",
                error=outcome.error,
                benchmark_outcome="unscored",
                runtime_outcome="not_started",
                terminal_kind=outcome.terminal_kind,
                started_at=started.isoformat(),
                ended_at=ended.isoformat(),
                wall_time_sec=wall_time,
            )
        )
        store.append_event(
            "cell_state",
            cell=cell,
            status="failed",
            message=outcome.error,
            terminal_kind=outcome.terminal_kind,
            wall_time_sec=wall_time,
        )
        return execution_env, True, outcome


def _host_process_command(command: Sequence[str]) -> list[str]:
    """Resolve host executables without changing the approved logical command."""

    resolved = list(command)
    if resolved and resolved[0] == "harbor":
        harbor = resolve_console_script("harbor")
        if harbor is not None:
            resolved[0] = harbor
    return resolved


def _run_cell_process(
    cell: PlannedCell,
    repo_root: Path,
    store: _RunStore,
    env: Mapping[str, str],
    *,
    cancellation_event: threading.Event | None = None,
) -> int:
    if cell.execution_kind == "agent":
        verify_harbor_job_attestation(cell.config_path, repo_root)
    physical_suffix = (
        f"-{cell.physical_execution_id[:12]}" if cell.physical_execution_id else ""
    )
    log_path = store.logs_dir / f"{cell.id}{physical_suffix}.log"
    command = _host_process_command(cell.command)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(env),
            cwd=repo_root,
            start_new_session=True,
        )
    except OSError as exc:
        raise TypedInfrastructureFailure(
            "runner_start_failure",
            f"runner process could not start: {exc}",
        ) from exc
    store.append_cell(
        cell.record(
            "running",
            harbor_pid=process.pid,
            harbor_process_group=process.pid,
        )
    )
    secrets = secrets_from_env(env)
    reader_error: list[BaseException] = []
    deadline = (
        time.monotonic() + cell.outer_wall_time_sec
        if cell.outer_wall_time_sec is not None
        else None
    )
    timed_out = False

    with log_path.open("a") as log:
        assert process.stdout is not None
        stdout = process.stdout

        def drain_output() -> None:
            try:
                with stdout:
                    for line in stdout:
                        safe_line = redact_text(line, secrets)
                        log.write(safe_line)
                        log.flush()
                        print(safe_line, end="", flush=True)
                        store.append_event("log", cell=cell, chunk=safe_line)
            except BaseException as exc:  # pragma: no cover - defensive I/O guard
                reader_error.append(exc)

        reader = threading.Thread(
            target=drain_output,
            name=f"fugue-cell-log-{cell.id}",
            daemon=True,
        )
        reader.start()
        while process.poll() is None:
            if cancellation_event is not None and cancellation_event.wait(0.1):
                _terminate_process_group(process)
                break
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                _terminate_process_group(process)
                break
            time.sleep(0.05)
        returncode = process.wait()
        reader.join(timeout=2)
        if reader.is_alive():  # pragma: no cover - dead children should close the pipe
            stdout.close()
            reader.join(timeout=2)
    if reader_error:
        raise RuntimeError(f"cell log reader failed: {reader_error[0]}")
    if timed_out:
        raise _CellWallTimeExceeded(
            "outer cell wall-time limit exceeded after "
            f"{cell.outer_wall_time_sec} seconds"
        )
    return returncode


def _verify_cell_inputs(cell: PlannedCell, repo_root: Path) -> None:
    config_path = cell.config_path
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if cell.config_sha256 and _path_digest(config_path) != cell.config_sha256:
        raise RuntimeError(f"config drift: {config_path}")
    for raw_path, expected in cell.runtime_assets:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        if _path_digest(path) != expected:
            raise RuntimeError(f"runtime asset drift: {path}")


def _path_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"cannot read immutable input {path}: {exc}") from exc


def _harbor_job_result(cell: PlannedCell, repo_root: Path) -> _HarborJobResult:
    path = cell.result_path
    if not path.is_absolute():
        path = repo_root / path
    try:
        result = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        return _HarborJobResult(
            f"Harbor did not produce a readable job result: {exc}", "unscored"
        )
    bound_terminal = _bound_harbor_terminal_result(
        cell=cell,
        repo_root=repo_root,
        raw_job=result,
    )
    if bound_terminal is not None:
        return bound_terminal
    stats = result.get("stats") or {}
    errored = int(stats.get("n_errored_trials") or 0)
    cancelled = int(stats.get("n_cancelled_trials") or 0)
    if cell.physical_execution_id and errored:
        return _HarborJobResult(
            "durable Harbor trial error lacks structured terminal evidence",
            "unscored",
            terminal_kind="evidence_failure",
            runtime_outcome="not_started",
        )
    if errored:
        return _HarborJobResult(f"{errored} Harbor trial(s) errored", "unscored")
    if cancelled:
        return _HarborJobResult(
            f"{cancelled} Harbor trial(s) were cancelled", "unscored"
        )
    rewards: list[float] = []
    for evaluation in (stats.get("evals") or {}).values():
        reward_buckets = ((evaluation or {}).get("reward_stats") or {}).get(
            "reward"
        ) or {}
        for raw_reward, trial_ids in reward_buckets.items():
            try:
                reward = float(raw_reward)
            except (TypeError, ValueError):
                return _HarborJobResult(
                    f"Harbor job result contains an invalid reward: {raw_reward!r}",
                    "unscored",
                )
            if not math.isfinite(reward):
                return _HarborJobResult(
                    f"Harbor job result contains a non-finite reward: {raw_reward!r}",
                    "unscored",
                )
            count = len(trial_ids) if isinstance(trial_ids, list) else 1
            rewards.extend([reward] * count)
    if not rewards:
        return _HarborJobResult(None, "unscored")
    if len(rewards) != 1:
        return _HarborJobResult(
            f"Harbor job result contains {len(rewards)} rewards for one cell",
            "unscored",
        )
    reward = rewards[0]
    return _HarborJobResult(
        None,
        "passed" if reward == 1.0 else "failed",
        reward,
    )


def _bound_harbor_terminal_result(
    *,
    cell: PlannedCell,
    repo_root: Path,
    raw_job: Mapping[str, Any],
) -> _HarborJobResult | None:
    if not cell.physical_execution_id:
        return None
    physical_root = (
        repo_root
        / ".fugue"
        / "runtime"
        / cell.run_id
        / "physical-executions"
        / cell.physical_execution_id
    ).resolve()
    receipt_path = physical_root / "harbor-terminal.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, Mapping):
        return None
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if (
        receipt.get("receipt_digest") != stable_digest(unsigned)
        or receipt.get("classifier_digest")
        != HARBOR_TERMINAL_CLASSIFIER_DIGEST
        or receipt.get("logical_attempt_id") != cell.attempt_id
        or receipt.get("physical_execution_id") != cell.physical_execution_id
        or receipt.get("retry_ordinal") != cell.retry_ordinal
        or receipt.get("config_sha256") != cell.config_sha256
        or receipt.get("source_result_path")
        != _absolute_cell_path(cell.result_path, repo_root).resolve().as_posix()
    ):
        return None
    raw_trial_path = receipt.get("terminal_trial_path")
    raw_snapshot_path = receipt.get("terminal_result_path")
    if not isinstance(raw_trial_path, str) or not isinstance(
        raw_snapshot_path, str
    ):
        return None
    trial_path = Path(raw_trial_path).resolve()
    snapshot_path = Path(raw_snapshot_path).resolve()
    try:
        trial_path.relative_to(physical_root)
        snapshot_path.relative_to(physical_root)
    except ValueError:
        return None
    result_path = _absolute_cell_path(cell.result_path, repo_root)
    if any(
        not candidate.is_file() or candidate.is_symlink()
        for candidate in (trial_path, snapshot_path, result_path)
    ):
        return None
    if (
        hashlib.sha256(trial_path.read_bytes()).hexdigest()
        != receipt.get("terminal_trial_sha256")
        or hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        != receipt.get("terminal_result_sha256")
    ):
        return None
    try:
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
        snapshot_job = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(trial, Mapping) or not isinstance(snapshot_job, Mapping):
        return None
    try:
        snapshot_outcome = classify_harbor_terminal(
            snapshot_job,
            cell_id=cell.id,
            trial=trial,
        )
        outcome = classify_harbor_terminal(
            raw_job,
            cell_id=cell.id,
            trial=trial,
        )
    except (TypeError, ValueError):
        return None
    if receipt.get("cell_outcome") != snapshot_outcome or outcome != snapshot_outcome:
        return None
    return _HarborJobResult(
        error=str(outcome["error"]) if outcome.get("error") is not None else None,
        benchmark_outcome=str(outcome["benchmark_outcome"]),  # type: ignore[arg-type]
        reward=(
            float(outcome["reward"]) if outcome.get("reward") is not None else None
        ),
        terminal_kind=str(outcome["terminal_kind"]),  # type: ignore[arg-type]
        runtime_outcome=str(outcome["runtime_outcome"]),  # type: ignore[arg-type]
    )


def _absolute_cell_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _write_physical_runner_terminal_observation(
    *,
    repo_root: Path,
    cell: PlannedCell,
    outcome: CellOutcome,
) -> Path:
    """Persist the runner exit observation before host-only finalization."""

    result_path = cell.result_path
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    result_reference = None
    result_sha256 = None
    if result_path.is_file() and not result_path.is_symlink():
        result_reference = result_path.resolve().relative_to(
            repo_root.resolve()
        ).as_posix()
        result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    unsigned = {
        "schema_version": 1,
        "logical_attempt_id": cell.attempt_id,
        "physical_execution_id": cell.physical_execution_id,
        "retry_ordinal": cell.retry_ordinal,
        "config_sha256": cell.config_sha256,
        "result_reference": result_reference,
        "result_sha256": result_sha256,
        "cell_outcome": {
            "cell_id": outcome.cell_id,
            "status": outcome.status,
            "returncode": outcome.returncode,
            "error": outcome.error,
            "benchmark_outcome": outcome.benchmark_outcome,
            "reward": outcome.reward,
            "runtime_outcome": outcome.runtime_outcome,
            "terminal_kind": outcome.terminal_kind,
        },
    }
    payload = {**unsigned, "receipt_digest": stable_digest(unsigned)}
    path = (
        repo_root
        / ".fugue"
        / "runtime"
        / cell.run_id
        / "physical-executions"
        / cell.physical_execution_id
        / "runner-terminal.json"
    )
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("physical runner terminal receipt changed")
    else:
        atomic_write_json(path, payload)
    return path


def update_run_manifest(
    repo_root: Path,
    run_id: str,
    updater: Callable[[dict[str, Any]], dict[str, Any]],
) -> Path:
    path = repo_root / ".fugue" / "runtime" / run_id / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(".lock").as_posix()):
        existing = read_run_manifest(path.parent) or {}
        values = updater(dict(existing))
        created_at = existing.get("created_at") or datetime.now(UTC).isoformat()
        atomic_write_json(
            path,
            {
                **existing,
                "schema_version": 1,
                "run_id": run_id,
                "created_at": created_at,
                "updated_at": datetime.now(UTC).isoformat(),
                **values,
            },
        )
    return path


def write_run_manifest(repo_root: Path, run_id: str, values: dict[str, Any]) -> Path:
    return update_run_manifest(repo_root, run_id, lambda _: values)


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
    status: Literal["failed", "cancelled", "interrupted"],
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
            "benchmark_outcome": "failed" if status == "failed" else "unscored",
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
    return latest_jsonl_records(path, "cell_id")


def record_recovered_cell_outcome(
    repo_root: Path,
    cell: PlannedCell,
    outcome: CellOutcome,
) -> None:
    """Idempotently repair the canonical run ledger from durable recovery state."""

    run_dir = repo_root / ".fugue" / "runtime" / cell.run_id
    latest = {
        str(item.get("cell_id") or ""): item
        for item in latest_cell_records(run_dir / "cells.jsonl")
    }.get(cell.id)
    if (
        latest is not None
        and latest.get("physical_execution_id") == cell.physical_execution_id
        and latest.get("terminal_kind") == outcome.terminal_kind
        and latest.get("status") == outcome.status
    ):
        return
    store = _RunStore(run_dir)
    ended = datetime.now(UTC).isoformat()
    store.append_cell(
        cell.record(
            outcome.status,
            returncode=outcome.returncode,
            error=outcome.error,
            benchmark_outcome=outcome.benchmark_outcome,
            reward=outcome.reward,
            runtime_outcome=outcome.runtime_outcome,
            terminal_kind=outcome.terminal_kind,
            recovered=True,
            ended_at=ended,
        )
    )
    store.append_event(
        "cell_state_recovered",
        cell=cell,
        status=outcome.status,
        message=outcome.error,
        terminal_kind=outcome.terminal_kind,
    )


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
