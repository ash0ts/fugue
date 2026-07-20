from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock

from fugue.bench.campaign_contracts import CampaignError
from fugue.bench.files import atomic_write_json
from fugue.bench.library import validate_id


@dataclass(frozen=True)
class CampaignStore:
    """Atomic local storage and lock layout for one Fugue repository."""

    repo_root: Path
    runtime_dir: Path

    def campaign_dir(self, campaign_id: str) -> Path:
        validate_id(campaign_id, kind="campaign id")
        return self.repo_root / self.runtime_dir / campaign_id

    def campaign_lock(self, campaign_id: str) -> FileLock:
        root = self.campaign_dir(campaign_id)
        root.mkdir(parents=True, exist_ok=True)
        return FileLock((root / ".campaign.lock").as_posix())

    def operation_lock(self, campaign_id: str, operation_id: str) -> FileLock:
        root = self.campaign_dir(campaign_id) / "operations"
        root.mkdir(parents=True, exist_ok=True)
        return FileLock((root / f"{operation_id}.lock").as_posix())

    def run_lock(self, campaign_id: str, run_id: str) -> FileLock:
        validate_id(run_id, kind="run id")
        root = self.campaign_dir(campaign_id) / "runs"
        root.mkdir(parents=True, exist_ok=True)
        return FileLock((root / f"{run_id}.lock").as_posix())

    def write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        atomic_write_json(path, value)

    def append_json(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: row {number} must be an object")
        rows.append(value)
    return rows


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_last_json_object(path: Path) -> dict[str, Any] | None:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell() - 1
        while position >= 0:
            handle.seek(position)
            if handle.read(1) not in {b"\n", b"\r"}:
                break
            position -= 1
        if position < 0:
            return None
        end = position + 1
        while position >= 0:
            handle.seek(position)
            if handle.read(1) == b"\n":
                position += 1
                break
            position -= 1
        start = max(0, position)
        handle.seek(start)
        raw = handle.read(end - start).decode("utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CampaignError(
            "event_log_corrupt",
            "the final campaign event is invalid JSON",
            category="evidence",
        ) from exc
    if not isinstance(value, dict):
        raise CampaignError(
            "event_log_corrupt",
            "the final campaign event is not an object",
            category="evidence",
        )
    return value


def event_id_in_log(path: Path, event_id: str) -> bool:
    """Rare recovery scan used only when an append outlived its index write."""

    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CampaignError(
                    "event_log_corrupt",
                    f"campaign event {number} is invalid JSON",
                    category="evidence",
                ) from exc
            if isinstance(value, dict) and value.get("event_id") == event_id:
                return True
    return False


def sha256_path(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
