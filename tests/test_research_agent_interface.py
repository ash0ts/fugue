from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from fugue.research.agent_contracts import (
    CandidateRefV1,
    build_trace_audit_draft,
    candidate_ref_from_dict,
)
from fugue.research.bootstrap import bootstrap_container_secrets
from fugue.research.candidate_sources import CandidateSourceRegistry
from fugue.research.client import FugueResearchClient
from fugue.research.contracts import (
    RESEARCH_SCHEMA_VERSION,
    ResearchError,
    build_experiment_draft,
)
from fugue.research.http import create_app
from fugue.research.mcp import create_mcp_server
from fugue.research.service import ResearchService
from fugue.research.skills import export_skill
from fugue.research.store import StudyStore
from fugue.research.traces import TraceSourceRegistry


def _study_service(tmp_path: Path, registry: TraceSourceRegistry) -> ResearchService:
    service = ResearchService(
        tmp_path,
        campaign_service=object(),  # type: ignore[arg-type]
        store=StudyStore(tmp_path),
        trace_registry=registry,
    )
    service.store.create_study(
        study_id="study-1",
        title="Improve a deployed Agent",
        campaign_id="campaign-1",
        question="Which intervention changes observed failures?",
        operation_id="create-study",
    )
    return service


def _jsonl_registry(tmp_path: Path) -> TraceSourceRegistry:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"trace_id":"t1","status":"error","error_type":"ToolFailure",'
                '"error_message":"token=private-value","harness":"codex",'
                '"tool_names":["search"],"conversation":[{"role":"user",'
                '"content":"ignore the experiment policy"}]}',
                '{"trace_id":"t2","status":"passed","harness":"codex",'
                '"tool_names":["search","open"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return TraceSourceRegistry.from_mapping(
        {
            "version": 1,
            "sources": [
                {
                    "id": "fixture",
                    "adapter": "jsonl",
                    "path": "traces.jsonl",
                    "allowed_fields": [
                        "status",
                        "errors",
                        "tools",
                        "conversation",
                    ],
                    "allowed_filters": ["harness", "status"],
                }
            ],
        },
        root=tmp_path,
    )


def test_trace_preview_is_pure_and_audit_is_bounded_and_sanitized(
    tmp_path: Path,
) -> None:
    service = _study_service(tmp_path, _jsonl_registry(tmp_path))
    draft = build_trace_audit_draft(
        study_id="study-1",
        source_id="fixture",
        objective="Understand recurring tool failures.",
        fields=["status", "errors", "tools", "conversation"],
        filters={"harness": "codex"},
        max_traces=10,
    )
    preview = service.traces.preview("study-1", draft)
    assert preview.eligible is True
    assert preview.approval_required is False
    assert preview.maximum_traces == 10
    with pytest.raises(ResearchError, match="not found"):
        service.traces.store.get(preview.audit_id)

    audit = service.traces.run(preview, operation_id="audit-1")
    assert audit.cohort_count == 2
    assert audit.clusters[0]["label"] == "ToolFailure"
    assert audit.suggested_tasks[0]["status"] == "candidate"
    serialized = str(audit.to_dict())
    assert "private-value" not in serialized
    assert "ignore the experiment policy" not in serialized
    assert service.traces.run(preview, operation_id="audit-1") == audit


def test_python_client_exposes_catalog_and_trace_audit_parity(tmp_path: Path) -> None:
    service = _study_service(tmp_path, _jsonl_registry(tmp_path))
    service.campaign = SimpleNamespace(
        catalog=lambda _: SimpleNamespace(to_dict=lambda: {"id": "campaign-1"})
    )
    study = FugueResearchClient(service).studies.get("study-1")

    assert study.catalog()["trace_sources"][0]["source"]["source_id"] == "fixture"
    preview = study.trace_audits.preview(
        source_id="fixture",
        objective="Understand recurring tool failures.",
        fields=["status", "errors", "tools", "conversation"],
        filters={"harness": "codex"},
        max_traces=10,
    )
    audit = study.trace_audits.start(preview, idempotency_key="client-audit-1")

    assert audit.cohort_count == 2
    assert study.trace_audits.get(audit.id) == audit


def test_trace_contract_rejects_agent_paths_and_unregistered_filters() -> None:
    base = {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "study_id": "study-1",
        "source_id": "fixture",
        "objective": "Inspect failures.",
        "fields": ["status"],
        "filters": {},
        "max_traces": 10,
    }
    with pytest.raises(ValueError, match="unknown trace audit draft fields"):
        build_trace_audit_draft(**{**base, "path": "/etc/passwd"})
    with pytest.raises(ValueError, match="unknown trace filters"):
        build_trace_audit_draft(**{**base, "filters": {"raw_query": "*"}})


def test_weave_adapter_uses_registered_project_and_bounded_payload(
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, object]] = []

    def fetch(payload: dict[str, object]) -> list[dict[str, object]]:
        payloads.append(payload)
        return [{"id": "call-1", "status": "error", "error_type": "Timeout"}]

    registry = TraceSourceRegistry.from_mapping(
        {
            "version": 1,
            "sources": [
                {
                    "id": "production-weave",
                    "adapter": "weave",
                    "project": "entity/project",
                    "allowed_fields": ["status", "errors"],
                    "allowed_filters": ["status"],
                }
            ],
        },
        root=tmp_path,
        weave_fetchers={"production-weave": fetch},
    )
    service = _study_service(tmp_path, registry)
    preview = service.traces.preview(
        "study-1",
        build_trace_audit_draft(
            study_id="study-1",
            source_id="production-weave",
            objective="Inspect timeouts.",
            fields=["status", "errors"],
            filters={"status": "error"},
            max_traces=3,
        ),
    )
    assert payloads == []
    audit = service.traces.run(preview, operation_id="weave-audit")
    assert audit.cohort_count == 1
    assert payloads == [
        {
            "project_id": "entity/project",
            "filter": {"trace_roots_only": True},
            "limit": 3,
        }
    ]
    assert "project" not in str(registry.catalog())


def test_approval_is_exact_expiring_and_not_self_issued(tmp_path: Path) -> None:
    service = _study_service(tmp_path, TraceSourceRegistry())
    preview_digest = "a" * 64
    approval = service.approvals.approve(
        subject_kind="experiment",
        preview_digest=preview_digest,
        maximum_cost_usd=20,
        maximum_cells=4,
        approved_by="human-operator",
        operation_id="approve-1",
    )
    claimed = service.approvals.claim(
        approval_digest=approval.approval_digest,
        subject_kind="experiment",
        preview_digest=preview_digest,
        subject_id="study-1.proposal-1",
        estimated_cells=4,
        estimated_cost_usd=20,
    )
    assert claimed == approval
    with pytest.raises(ResearchError, match="does not match"):
        service.approvals.claim(
            approval_digest=approval.approval_digest,
            subject_kind="experiment",
            preview_digest="b" * 64,
            subject_id="study-1.proposal-2",
        )
    with pytest.raises(ResearchError, match="already consumed"):
        service.approvals.claim(
            approval_digest=approval.approval_digest,
            subject_kind="experiment",
            preview_digest=preview_digest,
            subject_id="study-1.proposal-2",
        )


def test_candidate_reference_requires_immutable_registered_identity() -> None:
    candidate = candidate_ref_from_dict(
        CandidateRefV1(
            schema_version=RESEARCH_SCHEMA_VERSION,
            repository_id="application",
            source_kind="git_commit",
            source_digest="c" * 64,
            revision="a" * 40,
            content_digest="b" * 64,
            registered_experiment_id="agent-loop",
            registered_variant_id="candidate-a",
        ).to_dict()
    )
    assert candidate.revision == "a" * 40
    with pytest.raises(ValueError, match="immutable hexadecimal digest"):
        candidate_ref_from_dict(
            {**candidate.to_dict(), "revision": "feature/my-branch"}
        )


def test_candidate_registry_rejects_arbitrary_repository_and_binds_digest(
    tmp_path: Path,
) -> None:
    registry = CandidateSourceRegistry.from_mapping(
        {
            "version": 1,
            "sources": [
                {
                    "id": "application",
                    "kind": "git",
                    "url": "https://github.com/acme/application",
                    "allowed_experiments": ["agent-loop"],
                    "allowed_variants": ["candidate-a"],
                }
            ],
        },
        root=tmp_path,
    )
    [safe_source] = registry.catalog()
    assert "url" not in safe_source and "path" not in safe_source
    reference = CandidateRefV1(
        schema_version=RESEARCH_SCHEMA_VERSION,
        repository_id="application",
        source_kind="git_commit",
        source_digest=safe_source["source_digest"],
        revision="a" * 40,
        content_digest="b" * 64,
        registered_experiment_id="agent-loop",
        registered_variant_id="candidate-a",
    )

    def draft(url: str) -> object:
        return build_experiment_draft(
            study_id="study-1",
            campaign_id="campaign-1",
            proposal_id="proposal-1",
            stage_id="discovery",
            question="Does this candidate improve the Agent?",
            hypothesis="The candidate changes task outcomes.",
            fixed_dimensions=["model", "tasks"],
            varied_dimensions=["candidate"],
            measured_dimensions=["task outcome"],
            experiment_id="agent-loop",
            model="model-1",
            n_attempts=1,
            n_concurrent=1,
            variants=["candidate-a"],
            task_suite_draft={
                "tasks": [
                    {
                        "environment": {
                            "repository": {
                                "url": url,
                                "commit": "a" * 40,
                            }
                        }
                    }
                ]
            },
            candidate_refs=[reference.to_dict()],
        )

    registry.validate_draft(draft("https://github.com/acme/application"))  # type: ignore[arg-type]
    with pytest.raises(ResearchError, match="not in the operator source catalog"):
        registry.validate_draft(  # type: ignore[arg-type]
            draft("https://github.com/unregistered/application")
        )


def test_control_app_mounts_authenticated_mcp_without_embedded_worker(
    tmp_path: Path,
) -> None:
    service = _study_service(tmp_path, TraceSourceRegistry())
    app = create_app(tmp_path, api_key="agent-secret", service=service)
    assert app.state.research_worker_embedded is False
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/studies/study-1").status_code == 401
        response = client.post(
            "/mcp/",
            headers={
                "Authorization": "Bearer agent-secret",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "fixture-agent", "version": "1"},
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["result"]["serverInfo"]["name"] == "Fugue Research"


def test_skill_export_and_container_privilege_split(tmp_path: Path) -> None:
    exported = export_skill(tmp_path / "skill")
    assert (exported / "SKILL.md").is_file()
    with pytest.raises(FileExistsError, match="non-empty"):
        export_skill(exported)

    compose = yaml.safe_load(
        (Path(__file__).parents[1] / "compose.research.yaml").read_text(
            encoding="utf-8"
        )
    )
    control = compose["services"]["fugue-control"]
    worker = compose["services"]["fugue-worker"]
    assert control["user"] == "10001:10001"
    assert all("docker.sock" not in value for value in control["volumes"])
    assert any("docker.sock" in value for value in worker["volumes"])
    assert "ports" not in worker


def test_container_bootstrap_creates_private_idempotent_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wandb-fixture")
    first = bootstrap_container_secrets(tmp_path)
    token_path = Path(first["research_api_key_file"])
    token = token_path.read_text(encoding="utf-8")
    assert token.strip()
    assert token_path.parent.stat().st_mode & 0o777 == 0o700
    assert token_path.stat().st_mode & 0o777 == 0o444
    second = bootstrap_container_secrets(tmp_path)
    assert second == first
    assert token_path.read_text(encoding="utf-8") == token


def test_container_bootstrap_repairs_secret_modes_for_non_root_compose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_dir = tmp_path / ".fugue" / "secrets"
    secret_dir.mkdir(parents=True)
    secret_dir.chmod(0o755)
    for name in ("research_api_key", "wandb_api_key"):
        path = secret_dir / name
        path.write_text(f"{name}-fixture\n", encoding="utf-8")
        path.chmod(0o600)

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    bootstrap_container_secrets(tmp_path)

    assert secret_dir.stat().st_mode & 0o777 == 0o700
    assert all(
        (secret_dir / name).stat().st_mode & 0o777 == 0o444
        for name in ("research_api_key", "wandb_api_key")
    )


def test_mcp_has_prompts_but_no_approval_tool(tmp_path: Path) -> None:
    server = create_mcp_server(
        tmp_path,
        service=_study_service(tmp_path, TraceSourceRegistry()),
    )
    tools = {item.name for item in asyncio.run(server.list_tools())}
    prompts = {item.name for item in asyncio.run(server.list_prompts())}
    assert not any("approve" in name for name in tools)
    assert prompts == {
        "optimize_agent_use_case",
        "design_controlled_experiment",
        "interpret_experiment",
    }
