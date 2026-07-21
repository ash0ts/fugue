from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

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
        events = service.store.events(experiment_id, after=after, limit=limit + 1)
        terminal = record.state in TERMINAL_EXPERIMENT_STATES
        if events or terminal or time.monotonic() >= deadline:
            break
        time.sleep(min(_POLL_INTERVAL_SECONDS, deadline - time.monotonic()))

    visible = events[:limit]
    next_cursor = visible[-1].sequence if visible else after
    planned, queued, running, completed = _cell_counts(service, record)
    latest = service.store.latest_event(experiment_id)
    has_more = len(events) > limit
    recommended = _recommended_check_seconds(record.state, terminal, has_more)
    now = datetime.now(UTC)
    return ExperimentEventPageV1(
        experiment_id=experiment_id,
        events=tuple(item.to_dict() for item in visible),
        next_cursor=next_cursor,
        has_more=has_more,
        state=record.state,
        terminal=terminal,
        planned_cells=planned,
        queued_cells=queued,
        running_cells=running,
        terminal_cells=completed,
        elapsed_seconds=max(0.0, (now - _timestamp(record.created_at)).total_seconds()),
        last_event_at=latest.created_at if latest else None,
        recommended_check_seconds=recommended,
        next_check_at=(now + timedelta(seconds=recommended)).isoformat(),
    )


def _cell_counts(service: ResearchService, record: object) -> tuple[int, int, int, int]:
    preview = getattr(record, "preview", {})
    planned = int(preview.get("estimated_cells") or 0)
    run_id = getattr(record, "run_id", None)
    if not run_id:
        terminal = getattr(record, "state", "") in TERMINAL_EXPERIMENT_STATES
        return planned, 0 if terminal else planned, 0, planned if terminal else 0
    try:
        cells = service.campaign.operator.run_summary(str(run_id)).cells
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
