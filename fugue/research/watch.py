from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fugue.research.contracts import TERMINAL_EXPERIMENT_STATES
from fugue.research.service import ResearchService

_MAX_WAIT_SECONDS = 30.0
_MAX_PAGE_SIZE = 200
_POLL_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True)
class ExperimentEventPageV1:
    experiment_id: str
    events: tuple[dict[str, object], ...]
    next_cursor: int
    has_more: bool
    state: str
    terminal: bool
    planned_cells: int
    queued_cells: int
    running_cells: int
    terminal_cells: int
    elapsed_seconds: float
    last_event_at: str | None
    recommended_check_seconds: int
    next_check_at: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def watch_experiment_page(
    service: ResearchService,
    experiment_id: str,
    *,
    after: int = 0,
    wait_seconds: float = 0.0,
    limit: int = 100,
) -> ExperimentEventPageV1:
    if after < 0:
        raise ValueError("after must be non-negative")
    if wait_seconds < 0 or wait_seconds > _MAX_WAIT_SECONDS:
        raise ValueError("wait_seconds must be between 0 and 30")
    if limit < 1 or limit > _MAX_PAGE_SIZE:
        raise ValueError("limit must be between 1 and 200")

    deadline = time.monotonic() + wait_seconds
    while True:
        record = service.store.get_experiment(experiment_id)
        timeline = _experiment_timeline(service, record)
        events = timeline[after : after + limit + 1]
        terminal = record.state in TERMINAL_EXPERIMENT_STATES
        if events or terminal or time.monotonic() >= deadline:
            break
        time.sleep(min(_POLL_INTERVAL_SECONDS, deadline - time.monotonic()))

    visible = events[:limit]
    next_cursor = after + len(visible)
    planned, queued, running, completed = _cell_counts(service, record)
    has_more = len(events) > limit
    recommended = _recommended_check_seconds(record.state, terminal, has_more)
    now = datetime.now(UTC)
    return ExperimentEventPageV1(
        experiment_id=experiment_id,
        events=tuple(visible),
        next_cursor=next_cursor,
        has_more=has_more,
        state=record.state,
        terminal=terminal,
        planned_cells=planned,
        queued_cells=queued,
        running_cells=running,
        terminal_cells=completed,
        elapsed_seconds=max(0.0, (now - _timestamp(record.created_at)).total_seconds()),
        last_event_at=(
            str(timeline[-1].get("created_at")) if timeline else None
        ),
        recommended_check_seconds=recommended,
        next_check_at=(now + timedelta(seconds=recommended)).isoformat(),
    )


def _experiment_timeline(
    service: ResearchService,
    record: object,
) -> list[dict[str, object]]:
    typed_record: Any = record
    experiment_id = str(typed_record.id)
    study_events = [
        item.to_dict()
        for item in service.store.events(experiment_id, after=0, limit=1000)
    ]
    run_events = _run_cell_events(service, record)
    combined = [*study_events, *run_events]
    combined.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            0 if item.get("event_type") != "cell_state" else 1,
            str(item.get("event_id") or ""),
        )
    )
    return [
        {**item, "sequence": sequence}
        for sequence, item in enumerate(combined, start=1)
    ]


def _run_cell_events(
    service: ResearchService,
    record: object,
) -> list[dict[str, object]]:
    typed_record: Any = record
    run_id = str(getattr(record, "run_id", None) or "")
    if not run_id:
        return []
    try:
        repo_root = Path(service.campaign.operator.repo_root)
    except AttributeError:
        return []
    run_dir = repo_root / ".fugue" / "runtime" / run_id
    cells = _jsonl(run_dir / "cells.jsonl")
    metadata: dict[str, dict[str, Any]] = {}
    for row in cells:
        cell_id = str(row.get("cell_id") or "")
        if cell_id and cell_id not in metadata:
            metadata[cell_id] = row
    result: list[dict[str, object]] = []
    for row in _jsonl(run_dir / "events.jsonl"):
        if row.get("event") != "cell_state":
            continue
        cell_id = str(row.get("cell_id") or "")
        cell = metadata.get(cell_id, {})
        created_at = str(row.get("recorded_at") or "")
        status = str(row.get("status") or "unknown")
        safe: dict[str, object] = {
            "schema_version": 1,
            "event_id": str(row.get("event_id") or ""),
            "study_id": str(typed_record.study_id),
            "experiment_id": str(typed_record.id),
            "run_id": run_id,
            "cell_id": cell_id,
            "state": status,
            "event_type": "cell_state",
            "message": f"Experiment cell entered {status} state.",
            "created_at": created_at,
            "task_id": str(cell.get("task_id") or ""),
            "harness": str(cell.get("harness") or ""),
            "variant_id": str(cell.get("variant_id") or ""),
            "trial_index": int(cell.get("trial_index") or 1),
        }
        for key in ("benchmark_outcome", "reward", "wall_time_sec"):
            if row.get(key) is not None:
                safe[key] = row[key]
        safe["event_digest"] = hashlib.sha256(
            json.dumps(safe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        result.append(safe)
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            # A worker may be appending the last JSONL record while the watch
            # page is read. Earlier complete events remain replayable and the
            # partial tail becomes visible on the next bounded check.
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _cell_counts(service: ResearchService, record: object) -> tuple[int, int, int, int]:
    preview = getattr(record, "preview", {})
    planned = int(preview.get("estimated_cells") or 0)
    run_id = getattr(record, "run_id", None)
    if not run_id:
        terminal = getattr(record, "state", "") in TERMINAL_EXPERIMENT_STATES
        return planned, 0 if terminal else planned, 0, planned if terminal else 0
    try:
        cells = service.campaign.operator.run_summary(str(run_id), recover=False).cells
    except (FileNotFoundError, RuntimeError, ValueError):
        return planned, planned, 0, 0
    running = sum(cell.status == "running" for cell in cells)
    queued = sum(cell.status == "pending" for cell in cells)
    terminal = sum(cell.status not in {"pending", "running"} for cell in cells)
    return planned, queued, running, terminal


def _recommended_check_seconds(state: str, terminal: bool, has_more: bool) -> int:
    if terminal or has_more:
        return 0
    if state == "running":
        return 30
    if state in {"preparing", "admitting", "launching"}:
        return 10
    return 5


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
