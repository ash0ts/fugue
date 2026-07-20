from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fugue.research.contracts import (
    attribution_from_dict,
    experiment_draft_from_dict,
    experiment_preview_from_dict,
    study_update_from_dict,
)
from fugue.research.service import ExperimentHandle, ResearchService, ResearchWorker


def create_mcp_server(
    repo_root: str | Path,
    *,
    env_file: str | Path | None = None,
    service: ResearchService | None = None,
) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "install fugue[research] to use the research MCP server"
        ) from exc

    research = service or ResearchService(
        Path(repo_root), Path(env_file) if env_file is not None else None
    )
    worker = ResearchWorker(research, poll_interval=0.5).start()
    mcp = FastMCP("Fugue Research")

    @mcp.tool()
    def create_study(
        study_id: str,
        title: str,
        campaign_id: str,
        question: str,
        idempotency_key: str,
        background: str = "",
        attribution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return research.store.create_study(
            study_id=study_id,
            title=title,
            campaign_id=campaign_id,
            question=question,
            background=background,
            attribution=attribution_from_dict(attribution or {}),
            operation_id=idempotency_key,
        ).to_dict()

    @mcp.tool()
    def read_study_context(study_id: str, max_chars: int = 32000) -> dict[str, Any]:
        return research.store.context(study_id, max_chars=max_chars).to_dict()

    @mcp.tool()
    def record_study_update(
        study_id: str,
        update: dict[str, Any],
        idempotency_key: str,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return research.store.update_study(
            study_id,
            study_update_from_dict(update),
            operation_id=idempotency_key,
            expected_revision=expected_revision,
        ).to_dict()

    @mcp.tool()
    def preview_experiment(study_id: str, draft: dict[str, Any]) -> dict[str, Any]:
        return research.preview_experiment(
            study_id,
            experiment_draft_from_dict(draft, require_digest=False),
        ).to_dict()

    @mcp.tool()
    def start_experiment(
        preview: dict[str, Any], idempotency_key: str
    ) -> dict[str, Any]:
        return research.start_experiment(
            experiment_preview_from_dict(preview),
            idempotency_key=idempotency_key,
        ).to_dict()

    @mcp.tool()
    def inspect_experiment(experiment_id: str) -> dict[str, Any]:
        record = research.store.get_experiment(experiment_id)
        return {
            "record": record.to_dict(),
            "events": [item.to_dict() for item in research.store.events(experiment_id)],
        }

    @mcp.tool()
    def cancel_experiment(
        experiment_id: str, reason: str, idempotency_key: str
    ) -> dict[str, Any]:
        return research.cancel_experiment(
            experiment_id,
            idempotency_key=idempotency_key,
            reason=reason,
        ).to_dict()

    @mcp.resource("fugue://studies/{study_id}/context")
    def study_context(study_id: str) -> str:
        return json.dumps(research.store.context(study_id).to_dict(), indent=2)

    @mcp.resource("fugue://experiments/{experiment_id}")
    def experiment_status(experiment_id: str) -> str:
        return json.dumps(
            research.store.get_experiment(experiment_id).to_dict(), indent=2
        )

    @mcp.resource("fugue://experiments/{experiment_id}/outcome")
    def experiment_outcome(experiment_id: str) -> str:
        return json.dumps(ExperimentHandle(research, experiment_id).result(), indent=2)

    mcp._fugue_worker = worker
    return mcp


def run(repo_root: str | Path, *, env_file: str | Path | None = None) -> None:
    create_mcp_server(repo_root, env_file=env_file).run()
