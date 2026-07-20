from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

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


def test_mcp_exposes_only_high_level_research_operations(tmp_path: Path) -> None:
    server = create_mcp_server(tmp_path, service=_service(tmp_path))
    try:
        tools = asyncio.run(server.list_tools())
        names = {tool.name for tool in tools}
        assert names == {
            "create_study",
            "read_study_context",
            "record_study_update",
            "preview_experiment",
            "start_experiment",
            "inspect_experiment",
            "cancel_experiment",
        }
        templates = asyncio.run(server.list_resource_templates())
        assert len(templates) == 3
    finally:
        server._fugue_worker.stop()
