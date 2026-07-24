from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from fugue.redaction import redact_text, redact_value, secrets_from_env
from fugue.research.access import (
    ResearchAccessAuthorizer,
    access_scope,
    bind_current_grant,
    require_current_access,
    research_id_from_path,
    reset_current_grant,
)
from fugue.research.agent_contracts import (
    trace_audit_draft_from_dict,
    trace_audit_preview_from_dict,
)
from fugue.research.candidate_sources import CandidateSourceRegistry
from fugue.research.contracts import (
    ResearchError,
    experiment_draft_from_dict,
    experiment_preview_from_dict,
    study_update_from_dict,
)
from fugue.research.mcp import create_mcp_server
from fugue.research.service import ExperimentHandle, ResearchService
from fugue.research.task_recipes import task_recipe_draft_from_dict
from fugue.research.traces import TraceSourceRegistry
from fugue.research.watch import watch_experiment_page


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateStudyBody(StrictBody):
    study_id: str
    title: str
    campaign_id: str
    question: str
    background: str = ""
    parent_study_ids: tuple[str, ...] = ()
    attribution: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str


class CreateResearchBody(StrictBody):
    research_id: str
    title: str
    campaign_id: str
    question: str
    background: str = ""
    parent_research_ids: tuple[str, ...] = ()
    attribution: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str


class UpdateStudyBody(StrictBody):
    update: dict[str, Any]
    expected_revision: int | None = None
    idempotency_key: str


class PreviewAuditBody(StrictBody):
    draft: dict[str, Any]


class StartAuditBody(StrictBody):
    preview: dict[str, Any]
    approval_digest: str | None = None
    idempotency_key: str


class PreviewExperimentBody(StrictBody):
    draft: dict[str, Any]


class PreviewTaskRecipeBody(StrictBody):
    draft: dict[str, Any]


class StartExperimentBody(StrictBody):
    preview: dict[str, Any]
    approval_digest: str | None = None
    idempotency_key: str


class RequestStudyApprovalBody(StrictBody):
    preview: dict[str, Any]
    idempotency_key: str


class CancelExperimentBody(StrictBody):
    idempotency_key: str
    reason: str


def create_app(  # noqa: C901
    repo_root: str | Path,
    *,
    env_file: str | Path | None = None,
    api_key: str | None = None,
    max_request_bytes: int = 1_048_576,
    max_concurrent_requests: int = 32,
    allowed_hosts: tuple[str, ...] | None = None,
    access_grants: Path | None = None,
    authorizer: ResearchAccessAuthorizer | None = None,
    service: ResearchService | None = None,
    mount_mcp: bool = True,
) -> FastAPI:
    research = service or ResearchService(
        Path(repo_root),
        Path(env_file) if env_file is not None else None,
    )
    expected_key = api_key if api_key is not None else _secret("FUGUE_RESEARCH_API_KEY")
    grant_path = access_grants
    if grant_path is None and os.getenv("FUGUE_RESEARCH_ACCESS_GRANTS"):
        grant_path = Path(os.environ["FUGUE_RESEARCH_ACCESS_GRANTS"])
    access = authorizer or (
        ResearchAccessAuthorizer.from_file(grant_path)
        if grant_path is not None
        else ResearchAccessAuthorizer.for_token(expected_key)
    )
    mcp_app = None
    if mount_mcp:
        mcp = create_mcp_server(
            repo_root,
            service=research,
            streamable_http_path="/",
            authorize=require_current_access,
        )
        mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        if mcp_app is None:
            yield
            return
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(
        title="Fugue Research API",
        version="1.0.0",
        lifespan=lifespan,
    )
    trusted_hosts = allowed_hosts or _allowed_hosts()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(trusted_hosts))
    request_slots = asyncio.Semaphore(max(1, max_concurrent_requests))

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        if request_slots.locked():
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "server_busy"}},
                headers={"Retry-After": "1"},
            )
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > max_request_bytes):
            return JSONResponse(
                status_code=413,
                content={"error": {"code": "request_too_large"}},
            )
        body = b""
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            body = await request.body()
            if len(body) > max_request_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"code": "request_too_large"}},
                )
        if request.url.path not in {"/healthz", "/readyz"}:
            supplied = request.headers.get("authorization", "")
            prefix = "Bearer "
            token = supplied[len(prefix) :] if supplied.startswith(prefix) else ""
            grant = access.authorize(
                token,
                scope=access_scope(request.method, request.url.path),
                research_id=_request_research_id(request.url.path, body, research),
            )
            if grant is None:
                return JSONResponse(
                    status_code=403 if token else 401,
                    content={
                        "error": {"code": "forbidden" if token else "unauthorized"}
                    },
                    headers={"WWW-Authenticate": "Bearer"},
                )
            request.state.access_subject = grant.subject
            grant_token = bind_current_grant(grant)
        else:
            grant_token = None
        try:
            async with request_slots:
                return await call_next(request)
        finally:
            if grant_token is not None:
                reset_current_grant(grant_token)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(ResearchError)
    async def research_error_handler(_: Request, exc: ResearchError) -> JSONResponse:
        status = {
            "conflict": 409,
            "policy": 422,
            "admission": 422,
            "evidence": 409,
        }.get(exc.category, 400)
        return JSONResponse(status_code=status, content={"error": _safe_error(exc)})

    @app.exception_handler(ValueError)
    async def validation_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        error = ResearchError("invalid_request", str(exc))
        return JSONResponse(status_code=422, content={"error": _safe_error(error)})

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        _: Request,
        __: RequestValidationError,
    ) -> JSONResponse:
        error = ResearchError(
            "invalid_request",
            "the request body does not match the expected contract",
        )
        return JSONResponse(status_code=422, content={"error": _safe_error(error)})

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def ready() -> dict[str, Any]:
        with research.store._connect() as conn:
            conn.execute("SELECT 1").fetchone()
        return {"status": "ready"}

    @app.post("/v1/studies")
    def create_study(body: CreateStudyBody) -> dict[str, Any]:
        from fugue.research.contracts import attribution_from_dict

        return research.store.create_study(
            study_id=body.study_id,
            title=body.title,
            campaign_id=body.campaign_id,
            question=body.question,
            background=body.background,
            parent_study_ids=body.parent_study_ids,
            attribution=attribution_from_dict(body.attribution),
            operation_id=body.idempotency_key,
        ).to_dict()

    @app.post("/v1/research")
    def create_research(body: CreateResearchBody) -> dict[str, Any]:
        """Create programme-level Research while preserving stable V1 records."""

        from fugue.research.contracts import attribution_from_dict

        return research.store.create_study(
            study_id=body.research_id,
            title=body.title,
            campaign_id=body.campaign_id,
            question=body.question,
            background=body.background,
            parent_study_ids=body.parent_research_ids,
            attribution=attribution_from_dict(body.attribution),
            operation_id=body.idempotency_key,
        ).to_dict()

    @app.get("/v1/studies/{study_id}")
    def get_study(study_id: str) -> dict[str, Any]:
        return research.store.get_study(study_id).to_dict()

    @app.get("/v1/research/{research_id}")
    def get_research(research_id: str) -> dict[str, Any]:
        return research.store.get_study(research_id).to_dict()

    @app.get("/v1/studies/{study_id}/catalog")
    def get_catalog(study_id: str) -> dict[str, Any]:
        return research.catalog(study_id)

    @app.get("/v1/research/{research_id}/catalog")
    def get_research_catalog(research_id: str) -> dict[str, Any]:
        return research.catalog(research_id)

    @app.get("/v1/studies/{study_id}/context")
    def get_study_context(
        study_id: str,
        max_experiments: int = 20,
        max_results: int = 20,
        max_notes: int = 20,
        max_chars: int = 32000,
    ) -> dict[str, Any]:
        return research.store.context(
            study_id,
            max_experiments=max_experiments,
            max_results=max_results,
            max_notes=max_notes,
            max_chars=max_chars,
        ).to_dict()

    @app.get("/v1/research/{research_id}/context")
    def get_research_context(
        research_id: str,
        max_experiments: int = 20,
        max_results: int = 20,
        max_notes: int = 20,
        max_chars: int = 32000,
    ) -> dict[str, Any]:
        return research.store.context(
            research_id,
            max_experiments=max_experiments,
            max_results=max_results,
            max_notes=max_notes,
            max_chars=max_chars,
        ).to_dict()

    @app.post("/v1/studies/{study_id}/updates")
    def update_study(study_id: str, body: UpdateStudyBody) -> dict[str, Any]:
        return research.store.update_study(
            study_id,
            study_update_from_dict(body.update),
            operation_id=body.idempotency_key,
            expected_revision=body.expected_revision,
        ).to_dict()

    @app.post("/v1/research/{research_id}/updates")
    def update_research(research_id: str, body: UpdateStudyBody) -> dict[str, Any]:
        return research.store.update_study(
            research_id,
            study_update_from_dict(body.update),
            operation_id=body.idempotency_key,
            expected_revision=body.expected_revision,
        ).to_dict()

    @app.post("/v1/studies/{study_id}/trace-audits:preview")
    def preview_trace_audit(study_id: str, body: PreviewAuditBody) -> dict[str, Any]:
        return research.traces.preview(
            study_id,
            trace_audit_draft_from_dict(body.draft, require_digest=False),
        ).to_dict()

    @app.post("/v1/research/{research_id}/trace-audits:preview")
    def preview_research_trace_audit(
        research_id: str, body: PreviewAuditBody
    ) -> dict[str, Any]:
        return research.traces.preview(
            research_id,
            trace_audit_draft_from_dict(body.draft, require_digest=False),
        ).to_dict()

    @app.post("/v1/studies/{study_id}/trace-audits", status_code=201)
    def start_trace_audit(study_id: str, body: StartAuditBody) -> dict[str, Any]:
        preview = trace_audit_preview_from_dict(body.preview)
        if preview.study_id != study_id:
            raise ResearchError("study_mismatch", "audit belongs to another Study")
        return research.traces.run(
            preview,
            operation_id=body.idempotency_key,
            approval_digest=body.approval_digest,
        ).to_dict()

    @app.post("/v1/research/{research_id}/trace-audits", status_code=201)
    def start_research_trace_audit(
        research_id: str, body: StartAuditBody
    ) -> dict[str, Any]:
        preview = trace_audit_preview_from_dict(body.preview)
        if preview.study_id != research_id:
            raise ResearchError("study_mismatch", "audit belongs to another Research")
        return research.traces.run(
            preview,
            operation_id=body.idempotency_key,
            approval_digest=body.approval_digest,
        ).to_dict()

    @app.get("/v1/trace-audits/{audit_id}")
    def get_trace_audit(audit_id: str) -> dict[str, Any]:
        return research.traces.store.get(audit_id).to_dict()

    @app.post("/v1/research/{study_id}/task-suites:derive-preview")
    @app.post("/v1/studies/{study_id}/task-suites:derive-preview")
    def derive_task_suite_preview(
        study_id: str, body: PreviewTaskRecipeBody
    ) -> dict[str, Any]:
        """Compile selected trace provenance through one reviewed recipe."""

        return research.task_recipes.derive_preview(
            study_id,
            task_recipe_draft_from_dict(body.draft, require_digest=False),
        ).to_dict()

    @app.post("/v1/studies/{study_id}/experiments:preview")
    def preview_experiment(
        study_id: str, body: PreviewExperimentBody
    ) -> dict[str, Any]:
        return research.preview_experiment(
            study_id,
            experiment_draft_from_dict(body.draft, require_digest=False),
        ).to_dict()

    @app.post("/v1/research/{research_id}/studies:preview")
    def preview_controlled_study(
        research_id: str, body: PreviewExperimentBody
    ) -> dict[str, Any]:
        return research.preview_experiment(
            research_id,
            experiment_draft_from_dict(body.draft, require_digest=False),
        ).to_dict()

    @app.post("/v1/studies/{study_id}/experiments", status_code=202)
    def start_experiment(study_id: str, body: StartExperimentBody) -> dict[str, Any]:
        preview = experiment_preview_from_dict(body.preview)
        if preview.study_id != study_id:
            raise ResearchError("study_mismatch", "preview belongs to another Study")
        return research.start_experiment(
            preview,
            approval_digest=body.approval_digest,
            idempotency_key=body.idempotency_key,
        ).to_dict()

    @app.post("/v1/research/{research_id}/studies", status_code=202)
    def start_controlled_study(
        research_id: str, body: StartExperimentBody
    ) -> dict[str, Any]:
        preview = experiment_preview_from_dict(body.preview)
        if preview.study_id != research_id:
            raise ResearchError("study_mismatch", "preview belongs to another Research")
        return research.start_experiment(
            preview,
            approval_digest=body.approval_digest,
            idempotency_key=body.idempotency_key,
        ).to_dict()

    @app.post("/v1/research/{research_id}/studies:request-approval")
    def request_controlled_study_approval(
        research_id: str, body: RequestStudyApprovalBody
    ) -> dict[str, Any]:
        preview = experiment_preview_from_dict(body.preview)
        if preview.study_id != research_id:
            raise ResearchError("study_mismatch", "preview belongs to another Research")
        return research.request_study_approval(
            preview,
            idempotency_key=body.idempotency_key,
        ).to_dict()

    @app.get("/v1/research/{research_id}/studies:latest-approval-preview")
    def get_latest_controlled_study_approval_preview(
        research_id: str,
    ) -> dict[str, Any]:
        return research.latest_approval_preview(research_id).to_dict()

    @app.get("/v1/research-studies/{experiment_id}")
    @app.get("/v1/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        return research.store.get_experiment(experiment_id).to_dict()

    @app.get("/v1/research-studies/{experiment_id}/events")
    @app.get("/v1/experiments/{experiment_id}/events")
    async def experiment_events(
        experiment_id: str,
        request: Request,
        after: int = 0,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        cursor = max(after, int(last_event_id or 0))

        async def stream() -> AsyncIterator[str]:
            nonlocal cursor
            while not await request.is_disconnected():
                events = research.store.events(experiment_id, after=cursor)
                for event in events:
                    cursor = event.sequence
                    payload = json.dumps(event.to_dict(), separators=(",", ":"))
                    yield f"id: {cursor}\nevent: {event.event_type}\ndata: {payload}\n\n"
                record = research.store.get_experiment(experiment_id)
                if record.state in {"completed", "blocked", "cancelled", "interrupted"}:
                    return
                if not events:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/v1/research-studies/{experiment_id}/events:watch")
    @app.get("/v1/experiments/{experiment_id}/events:watch")
    def watch_experiment_events(
        experiment_id: str,
        after: int = 0,
        wait_seconds: float = 0.0,
        limit: int = 100,
    ) -> dict[str, object]:
        return watch_experiment_page(
            research,
            experiment_id,
            after=after,
            wait_seconds=wait_seconds,
            limit=limit,
        ).to_dict()

    @app.post("/v1/research-studies/{experiment_id}:cancel")
    @app.post("/v1/experiments/{experiment_id}:cancel")
    def cancel_experiment(
        experiment_id: str, body: CancelExperimentBody
    ) -> dict[str, Any]:
        return research.cancel_experiment(
            experiment_id,
            idempotency_key=body.idempotency_key,
            reason=body.reason,
        ).to_dict()

    @app.get("/v1/research-studies/{experiment_id}/outcome")
    @app.get("/v1/experiments/{experiment_id}/outcome")
    def get_outcome(experiment_id: str) -> dict[str, Any]:
        return ExperimentHandle(research, experiment_id).result()

    @app.get("/v1/research-publications")
    def research_publication_status() -> dict[str, Any]:
        return research.store.research_publication_status()

    if mcp_app is not None:
        app.mount("/mcp", mcp_app)
    app.state.research_service = research
    app.state.research_worker_embedded = False
    app.state.mcp_app = mcp_app
    return app


def serve(
    repo_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    api_key: str | None = None,
    env_file: str | Path | None = None,
    trace_sources: Path | None = None,
    candidate_sources: Path | None = None,
    access_grants: Path | None = None,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install fugue[research] to serve the research API") from exc
    root = Path(repo_root)
    resolved_key = api_key if api_key is not None else _secret("FUGUE_RESEARCH_API_KEY")
    if not resolved_key:
        raise RuntimeError("a research API key is required")
    configured_sources = trace_sources
    if configured_sources is None and os.getenv("FUGUE_RESEARCH_TRACE_SOURCES"):
        configured_sources = Path(os.environ["FUGUE_RESEARCH_TRACE_SOURCES"])
    registry = TraceSourceRegistry.from_file(configured_sources, env=os.environ)
    configured_candidates = candidate_sources
    if configured_candidates is None and os.getenv("FUGUE_RESEARCH_CANDIDATE_SOURCES"):
        configured_candidates = Path(os.environ["FUGUE_RESEARCH_CANDIDATE_SOURCES"])
    candidates = CandidateSourceRegistry.from_file(configured_candidates)
    service = ResearchService(
        root,
        Path(env_file) if env_file is not None else None,
        trace_registry=registry,
        candidate_sources=candidates,
    )
    uvicorn.run(
        create_app(
            root,
            api_key=resolved_key,
            access_grants=access_grants,
            service=service,
        ),
        host=host,
        port=port,
        access_log=False,
    )


def _secret(name: str) -> str:
    file_value = os.getenv(f"{name}_FILE", "").strip()
    if file_value:
        return Path(file_value).read_text(encoding="utf-8").strip()
    return os.getenv(name, "").strip()


def _allowed_hosts() -> tuple[str, ...]:
    configured = os.getenv("FUGUE_RESEARCH_TRUSTED_HOSTS", "").strip()
    if configured:
        values = tuple(item.strip() for item in configured.split(",") if item.strip())
        if not values:
            raise RuntimeError("FUGUE_RESEARCH_TRUSTED_HOSTS is empty")
        return values
    return ("127.0.0.1", "localhost", "[::1]")


def _request_research_id(
    path: str,
    body: bytes,
    research: ResearchService,
) -> str | None:
    selected = research_id_from_path(path)
    if selected is not None:
        return selected
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 3 and parts[:2] in (
        ["v1", "experiments"],
        ["v1", "research-studies"],
    ):
        experiment_id = parts[2].split(":", 1)[0]
        try:
            return research.store.get_experiment(experiment_id).study_id
        except ResearchError:
            return None
    if len(parts) >= 3 and parts[:2] == ["v1", "trace-audits"]:
        try:
            return research.traces.store.get(parts[2]).study_id
        except ResearchError:
            return None
    if not body:
        return None
    try:
        raw = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    for container in (raw, raw.get("draft"), raw.get("preview"), raw.get("update")):
        if not isinstance(container, dict):
            continue
        for key in ("research_id", "study_id"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _safe_error(exc: ResearchError) -> dict[str, Any]:
    secrets = secrets_from_env(os.environ)
    details = redact_value(exc.details, secrets=secrets)
    if len(json.dumps(details, default=str)) > 4096:
        details = {"omitted": True}
    safe = ResearchError(
        exc.code,
        redact_text(str(exc), secrets)[:1000],
        category=exc.category,
        retryable=exc.retryable,
        details=details,
    )
    return safe.to_dict()
