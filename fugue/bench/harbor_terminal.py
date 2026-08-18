from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harbor.models.job.plugin import BaseJobPlugin

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.harbor_outcome import (
    HARBOR_TERMINAL_CLASSIFIER_DIGEST,
    classify_harbor_terminal,
)

_DIGEST = re.compile(r"[0-9a-f]{64}")


class DurableHarborTerminalPlugin(BaseJobPlugin):
    """Write one immutable Harbor-owned terminal receipt for a physical cell.

    The plugin runs in the trusted host Harbor process, never in the Agent
    trial.  Fugue attaches it only to one-trial physical executions.  Harbor's
    internal terminal hook runs before this hook, so the captured JobResult
    already has terminal counts even if the outer Fugue controller disappears
    before observing the Harbor process exit.
    """

    def __init__(
        self,
        *,
        logical_attempt_id: str,
        physical_execution_id: str,
        retry_ordinal: str,
        cell_id: str,
        config_path: str,
        config_sha256: str,
        result_path: str,
        receipt_path: str,
    ) -> None:
        self.logical_attempt_id = _exact_digest(
            logical_attempt_id, "logical attempt id"
        )
        self.physical_execution_id = _exact_digest(
            physical_execution_id, "physical execution id"
        )
        self.retry_ordinal = _nonnegative_int(retry_ordinal, "retry ordinal")
        if not cell_id:
            raise ValueError("cell id is required")
        self.cell_id = cell_id
        self.config_path = Path(config_path).resolve()
        self.config_sha256 = _exact_digest(config_sha256, "config digest")
        self.result_path = Path(result_path).resolve()
        self.receipt_path = Path(receipt_path).resolve()
        self.snapshot_path = self.receipt_path.with_name(
            "harbor-terminal-result.json"
        )
        self.trial_snapshot_path = self.receipt_path.with_name(
            "harbor-terminal-trial.json"
        )
        self._started = False

    async def on_job_start(self, job: Any) -> None:
        if len(job) != 1:
            raise ValueError("durable Fugue Harbor cells require exactly one trial")
        if any(
            path.is_symlink()
            for path in (
                self.config_path,
                self.receipt_path,
                self.snapshot_path,
                self.trial_snapshot_path,
            )
            if path.exists()
        ):
            raise ValueError("durable Harbor terminal paths cannot be symlinks")
        if not self.config_path.is_file():
            raise ValueError("durable Harbor config is unavailable")
        config_bytes = self.config_path.read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != self.config_sha256:
            raise ValueError("durable Harbor config digest changed")
        config = json.loads(config_bytes)
        namespace = (
            ((config.get("fugue") or {}).get("physical_execution") or {})
            if isinstance(config, Mapping)
            else {}
        )
        if (
            not isinstance(namespace, Mapping)
            or namespace.get("logical_attempt_id") != self.logical_attempt_id
            or namespace.get("physical_execution_id")
            != self.physical_execution_id
            or namespace.get("retry_ordinal") != self.retry_ordinal
            or Path(str(namespace.get("harbor_result_path") or "")).resolve()
            != self.result_path
        ):
            raise ValueError("durable Harbor plugin identity disagrees with config")
        job_result_path = Path(str(getattr(job, "_job_result_path", ""))).resolve()
        if job_result_path != self.result_path:
            raise ValueError("Harbor JobResult path disagrees with the physical lock")
        if any(
            path.exists()
            for path in (
                self.receipt_path,
                self.snapshot_path,
                self.trial_snapshot_path,
            )
        ):
            raise ValueError("durable Harbor terminal namespace is not fresh")
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        job.on_trial_ended(self._on_trial_ended)
        self._started = True

    async def _on_trial_ended(self, event: Any) -> None:
        try:
            self._write_terminal(event)
        except Exception:
            # A receipt failure must never rewrite a behavioral task outcome.
            # Fugue will fail closed because neither this receipt nor the
            # parent process receipt will reconcile after controller loss.
            return

    def _write_terminal(self, event: Any) -> None:
        if not self._started:
            raise ValueError("durable Harbor plugin was not started")
        trial = getattr(event, "result", None)
        if trial is None or getattr(trial, "finished_at", None) is None:
            raise ValueError("Harbor trial is not terminal")
        if not self.result_path.is_file() or self.result_path.is_symlink():
            raise ValueError("terminal Harbor JobResult is unavailable")
        job_bytes = self.result_path.read_bytes()
        raw_job = json.loads(job_bytes)
        trial_payload = trial.model_dump(mode="json")
        outcome = classify_harbor_terminal(
            raw_job,
            cell_id=self.cell_id,
            trial=trial_payload,
        )
        trial_bytes = json.dumps(
            trial_payload, sort_keys=True, separators=(",", ":")
        ).encode()
        _write_immutable_bytes(self.snapshot_path, job_bytes)
        _write_immutable_bytes(self.trial_snapshot_path, trial_bytes)
        unsigned = {
            "schema_version": 1,
            "kind": "harbor_single_trial_terminal",
            "classifier_digest": HARBOR_TERMINAL_CLASSIFIER_DIGEST,
            "logical_attempt_id": self.logical_attempt_id,
            "physical_execution_id": self.physical_execution_id,
            "retry_ordinal": self.retry_ordinal,
            "config_path": self.config_path.as_posix(),
            "config_sha256": self.config_sha256,
            "source_result_path": self.result_path.as_posix(),
            "terminal_result_path": self.snapshot_path.as_posix(),
            "terminal_result_sha256": hashlib.sha256(job_bytes).hexdigest(),
            "terminal_trial_path": self.trial_snapshot_path.as_posix(),
            "terminal_trial_sha256": hashlib.sha256(trial_bytes).hexdigest(),
            "trial_id": str(getattr(trial, "id", "")),
            "trial_name": str(getattr(trial, "trial_name", "")),
            "trial_finished_at": str(getattr(trial, "finished_at", "")),
            "cell_outcome": outcome,
        }
        atomic_write_json(
            self.receipt_path,
            {**unsigned, "receipt_digest": stable_digest(unsigned)},
            mode=0o600,
        )
        _fsync(self.receipt_path)

    async def on_job_end(self, job_result: Any) -> None:
        # The one-trial END hook is the earlier authoritative boundary.  This
        # method verifies that Harbor reached its normal aggregate end when the
        # controller remains alive, without rewriting the receipt.
        if getattr(job_result, "finished_at", None) is None:
            return
        if not self.receipt_path.is_file():
            return


def _write_immutable_bytes(path: Path, value: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != value:
            raise ValueError("durable Harbor terminal snapshot changed")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    _fsync(path)


def _fsync(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _exact_digest(value: str, label: str) -> str:
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be an exact digest")
    return value


def _nonnegative_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} cannot be negative")
    return parsed
