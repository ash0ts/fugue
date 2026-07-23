from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    mode: int = 0o600,
) -> Path:
    """Durably replace one JSON file without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def inspect_docker_image(image: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "image missing").strip())
    values = json.loads(result.stdout)
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise RuntimeError("docker image inspect returned invalid JSON")
    return values[0]


def docker_build_command(*arguments: str) -> list[str]:
    """Build a Docker command compatible with both legacy and current CLIs.

    Docker 23 added the ``--provenance`` flag to ``docker build``. The
    research worker deliberately supports older distro-packaged clients too;
    those clients do not emit provenance attestations and reject the flag
    before they contact an otherwise compatible daemon.
    """
    try:
        help_text = subprocess.check_output(
            ["docker", "build", "--help"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        help_text = ""
    command = ["docker", "build"]
    if "--provenance" in help_text:
        command.append("--provenance=false")
    command.extend(arguments)
    return command


def require_unique(
    values: Iterable[str], kind: str, source: Path | None = None
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        prefix = f"{source}: " if source else ""
        raise ValueError(
            f"{prefix}duplicate {kind} id(s): {', '.join(sorted(duplicates))}"
        )


def store_consistent(
    values: dict[str, Any], key: str, value: Any, *, error: str
) -> None:
    existing = values.setdefault(key, value)
    if existing != value:
        raise ValueError(error)


def as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping, got {type(value).__name__}")
    return dict(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"expected list, got {type(value).__name__}")
    return list(value)


def latest_jsonl_records(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get(key):
            latest[str(record[key])] = record
    return list(latest.values())


def terminate_process_group(
    process: subprocess.Popen[Any], *, grace_sec: float = 2.0
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def terminate_async_process_group(
    process: asyncio.subprocess.Process, *, grace_sec: float = 5.0
) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_sec)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
