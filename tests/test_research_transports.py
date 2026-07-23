from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from fugue.research.contracts import (
    RESEARCH_SCHEMA_VERSION,
    ExperimentPreviewV1,
    ExperimentRecordV1,
    build_experiment_draft,
    now,
    sign_preview,
    sign_record,
)
from fugue.research.http import create_app
from fugue.research.mcp import create_mcp_server
from fugue.research.service import ResearchService
from fugue.research.store import StudyStore
from fugue.research.watch import (
    _cell_counts,
    _recommended_check_seconds,
    _run_cell_events,
)

_A = "a" * 64
_B = "b" * 64


def _service(tmp_path: Path) -> ResearchService:
    service = ResearchService(
        tmp_path,
        campaign_service=object(),  # type: ignore[arg-type]
        store=StudyStore(tmp_path),
    )
    service.store.create_study(
        study_id="study-1",
        title="Research transport",
        campaign_id="campaign-1",
        question="Do all surfaces preserve the same artifact?",
        operation_id="create-study",
    )
    return service


def _terminal_experiment(service: ResearchService) -> None:
    draft = build_experiment_draft(
        study_id="study-1",
        campaign_id="campaign-1",
        proposal_id="proposal-1",
        stage_id="discovery",
        question="Does the harness matter?",
        hypothesis="Harnesses may differ.",
        fixed_dimensions=["model"],
        varied_dimensions=["harness"],
        measured_dimensions=["pass"],
        experiment_id="baseline",
        model="model-1",
        n_attempts=1,
        n_concurrent=1,
    )
    preview = sign_preview(
        ExperimentPreviewV1(
            schema_version=RESEARCH_SCHEMA_VERSION,
            study_id="study-1",
            experiment_id="study-1.proposal-1",
            campaign_id="campaign-1",
            catalog_digest=_A,
            policy_digest=_B,
            draft=draft.to_dict(),
            task_suite_preview=None,
            plan_receipt={"plan_digest": _A},
            estimated_cells=1,
            estimated_calls={},
            estimated_cost_usd=0.0,
            eligible=True,
            blockers=(),
        )
    )
    timestamp = now()
    record = sign_record(
        ExperimentRecordV1(
            schema_version=RESEARCH_SCHEMA_VERSION,
            id=preview.experiment_id,
            study_id="study-1",
            campaign_id="campaign-1",
            state="queued",
            draft=draft.to_dict(),
            preview=preview.to_dict(),
            approval=None,
            parent_experiment_ids=(),
            proposal=None,
            plan=preview.plan_receipt,
            task_suite_lock=None,
            prepared_plan=None,
            admission=None,
            run_id=None,
            outcome=None,
            evaluation=None,
            analysis=None,
            error=None,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    service.store.insert_experiment(record, operation_id="start-1", input_digest=_A)
    terminal = sign_record(replace(record, state="cancelled", updated_at=now()))
    service.store.update_experiment(
        terminal,
        event_type="experiment_cancelled",
        message="Cancelled before launch.",
        release=True,
    )


def test_http_auth_revision_and_sse_cursor(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _terminal_experiment(service)
    app = create_app(tmp_path, api_key="secret", service=service)
    with TestClient(app) as client:
        assert client.get("/v1/studies/study-1").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        study = client.get("/v1/studies/study-1", headers=headers)
        assert study.status_code == 200
        assert study.json() == service.store.get_study("study-1").to_dict()
        response = client.post(
            "/v1/studies/study-1/updates",
            headers=headers,
            json={
                "update": {"message": "transport note"},
                "expected_revision": study.json()["revision"],
                "idempotency_key": "http-note",
            },
        )
        assert response.status_code == 200
        events = client.get(
            "/v1/experiments/study-1.proposal-1/events",
            headers={**headers, "Last-Event-ID": "1"},
        )
        assert events.status_code == 200
        assert "id: 2" in events.text
        assert "id: 1" not in events.text

        page = client.get(
            "/v1/experiments/study-1.proposal-1/events:watch",
            headers=headers,
            params={"after": 0, "wait_seconds": 0, "limit": 1},
        )
        assert page.status_code == 200
        assert page.json()["next_cursor"] == 1
        assert page.json()["has_more"] is True
        assert page.json()["planned_cells"] == 1
        assert page.json()["terminal_cells"] == 1
        assert page.json()["terminal"] is True
        assert page.json()["recommended_check_seconds"] == 0
        assert page.json()["next_check_at"].endswith("+00:00")


def test_watch_page_validates_bounded_cursor_wait_and_limit(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _terminal_experiment(service)
    app = create_app(tmp_path, api_key="secret", service=service)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        for params in (
            {"after": -1},
            {"wait_seconds": 31},
            {"limit": 0},
            {"limit": 201},
        ):
            response = client.get(
                "/v1/experiments/study-1.proposal-1/events:watch",
                headers=headers,
                params=params,
            )
            assert response.status_code == 422


def test_watch_recommendation_drains_pages_then_backs_off() -> None:
    assert _recommended_check_seconds("running", False, True) == 0
    assert _recommended_check_seconds("running", False, False) == 30
    assert _recommended_check_seconds("preparing", False, False) == 10
    assert _recommended_check_seconds("completed", True, False) == 0


def test_http_research_study_aliases_preserve_artifacts(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _terminal_experiment(service)
    app = create_app(tmp_path, api_key="secret", service=service)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        legacy = client.get("/v1/studies/study-1", headers=headers)
        research = client.get("/v1/research/study-1", headers=headers)
        assert research.status_code == 200
        assert research.json() == legacy.json()

        legacy_study = client.get("/v1/experiments/study-1.proposal-1", headers=headers)
        controlled_study = client.get(
            "/v1/research-studies/study-1.proposal-1", headers=headers
        )
        assert controlled_study.status_code == 200
        assert controlled_study.json() == legacy_study.json()


def test_watch_reads_active_worker_progress_without_supervisor_recovery() -> None:
    class Operator:
        def run_summary(self, run_id: str, *, recover: bool = True) -> object:
            assert run_id == "run-1"
            assert recover is False
            return SimpleNamespace(
                cells=(
                    SimpleNamespace(status="pending"),
                    SimpleNamespace(status="running"),
                    SimpleNamespace(status="failed"),
                )
            )

    service = SimpleNamespace(campaign=SimpleNamespace(operator=Operator()))
    record = SimpleNamespace(
        preview={"estimated_cells": 3},
        run_id="run-1",
        state="running",
    )

    assert _cell_counts(service, record) == (3, 1, 1, 1)


def test_watch_replays_safe_timestamped_cell_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / ".fugue/runtime/run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "cells.jsonl").write_text(
        json.dumps(
            {
                "cell_id": "cell-1",
                "task_id": "poisoned-trace",
                "harness": "codex",
                "variant_id": "trust-boundary-loop",
                "trial_index": 1,
                "status": "pending",
            }
        )
        + "\n"
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "event": "cell_state",
                    "event_id": "pending-1",
                    "cell_id": "cell-1",
                    "status": "pending",
                    "recorded_at": "2026-07-21T00:00:01+00:00",
                },
                {
                    "event": "log",
                    "event_id": "private-log",
                    "cell_id": "cell-1",
                    "chunk": "do not replay agent output",
                    "recorded_at": "2026-07-21T00:00:02+00:00",
                },
                {
                    "event": "cell_state",
                    "event_id": "passed-1",
                    "cell_id": "cell-1",
                    "status": "passed",
                    "benchmark_outcome": "passed",
                    "reward": 1.0,
                    "wall_time_sec": 4.5,
                    "recorded_at": "2026-07-21T00:00:03+00:00",
                },
            )
        )
        + "\n"
    )
    service = SimpleNamespace(
        campaign=SimpleNamespace(
            operator=SimpleNamespace(repo_root=tmp_path),
        )
    )
    record = SimpleNamespace(
        id="study-1.proposal-1",
        study_id="study-1",
        run_id="run-1",
    )

    events = _run_cell_events(service, record)

    assert [event["state"] for event in events] == ["pending", "passed"]
    assert events[-1]["task_id"] == "poisoned-trace"
    assert events[-1]["harness"] == "codex"
    assert events[-1]["variant_id"] == "trust-boundary-loop"
    assert events[-1]["benchmark_outcome"] == "passed"
    assert all("chunk" not in event for event in events)


def test_mcp_exposes_only_high_level_research_operations(tmp_path: Path) -> None:
    server = create_mcp_server(tmp_path, service=_service(tmp_path))
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    assert names == {
        "fugue_catalog",
        "fugue_research_catalog",
        "fugue_research_context",
        "fugue_research_create",
        "fugue_research_record",
        "fugue_research_result_record",
        "fugue_research_task_suite_derive_preview",
        "fugue_study_create",
        "fugue_study_context",
        "fugue_study_record",
        "fugue_study_preview",
        "fugue_study_start",
        "fugue_study_get",
        "fugue_study_watch",
        "fugue_study_cancel",
        "fugue_trace_audit_preview",
        "fugue_trace_audit_start",
        "fugue_task_suite_derive_preview",
        "fugue_experiment_preview",
        "fugue_experiment_start",
        "fugue_experiment_get",
        "fugue_experiment_watch",
        "fugue_experiment_cancel",
        "fugue_result_record",
    }
    templates = asyncio.run(server.list_resource_templates())
    assert len(templates) == 7
