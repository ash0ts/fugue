from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fugue.research.agent_contracts import (
    trace_audit_draft_from_dict,
    trace_audit_preview_from_dict,
)
from fugue.research.contracts import (
    ResearchError,
    attribution_from_dict,
    experiment_draft_from_dict,
    experiment_preview_from_dict,
    study_update_from_dict,
)
from fugue.research.service import ExperimentHandle, ResearchService
from fugue.research.task_recipes import task_recipe_draft_from_dict
from fugue.research.watch import watch_experiment_page


def create_mcp_server(  # noqa: C901
    repo_root: str | Path,
    *,
    env_file: str | Path | None = None,
    service: ResearchService | None = None,
    streamable_http_path: str = "/mcp",
    allowed_hosts: tuple[str, ...] = (
        "127.0.0.1:*",
        "localhost:*",
        "fugue-control:*",
        "testserver",
    ),
) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.server.transport_security import TransportSecuritySettings
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "install fugue[research] to use the research MCP server"
        ) from exc

    research = service or ResearchService(
        Path(repo_root), Path(env_file) if env_file is not None else None
    )
    mcp = FastMCP(
        "Fugue Research",
        streamable_http_path=streamable_http_path,
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(allowed_hosts),
        ),
    )

    @mcp.tool()
    def fugue_catalog(study_id: str) -> dict[str, Any]:
        """Read the safe campaign and trace-source catalog for one Study."""

        return research.catalog(study_id)

    @mcp.tool()
    def fugue_study_create(
        study_id: str,
        title: str,
        campaign_id: str,
        question: str,
        idempotency_key: str,
        background: str = "",
        attribution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create durable research memory without starting an experiment."""

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
    def fugue_study_context(study_id: str, max_chars: int = 32000) -> dict[str, Any]:
        """Read bounded Study context with explicit omission counts."""

        return research.store.context(study_id, max_chars=max_chars).to_dict()

    @mcp.tool()
    def fugue_study_record(
        study_id: str,
        update: dict[str, Any],
        idempotency_key: str,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Append notes or sourced results without silently rewriting history."""

        return research.store.update_study(
            study_id,
            study_update_from_dict(update),
            operation_id=idempotency_key,
            expected_revision=expected_revision,
        ).to_dict()

    @mcp.tool()
    def fugue_trace_audit_preview(
        study_id: str, draft: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate a bounded trace audit without reading the trace source."""

        return research.traces.preview(
            study_id,
            trace_audit_draft_from_dict(draft, require_digest=False),
        ).to_dict()

    @mcp.tool()
    def fugue_trace_audit_start(
        preview: dict[str, Any],
        idempotency_key: str,
        approval_digest: str | None = None,
    ) -> dict[str, Any]:
        """Run an accepted audit; metered audit profiles require approval."""

        return research.traces.run(
            trace_audit_preview_from_dict(preview),
            operation_id=idempotency_key,
            approval_digest=approval_digest,
        ).to_dict()

    @mcp.tool()
    def fugue_task_suite_derive_preview(
        study_id: str, draft: dict[str, Any]
    ) -> dict[str, Any]:
        """Map selected traces to a reviewed synthetic task without execution."""

        return research.task_recipes.derive_preview(
            study_id,
            task_recipe_draft_from_dict(draft, require_digest=False),
        ).to_dict()

    @mcp.tool()
    def fugue_experiment_preview(
        study_id: str, draft: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve and estimate an experiment without preparing or running it."""

        return research.preview_experiment(
            study_id,
            experiment_draft_from_dict(draft, require_digest=False),
        ).to_dict()

    @mcp.tool()
    def fugue_experiment_start(
        preview: dict[str, Any],
        approval_digest: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Queue an exact preview only after separate operator approval."""

        return research.start_experiment(
            experiment_preview_from_dict(preview),
            approval_digest=approval_digest,
            idempotency_key=idempotency_key,
        ).to_dict()

    @mcp.tool()
    def fugue_experiment_get(experiment_id: str) -> dict[str, Any]:
        """Inspect current state and ordered durable events."""

        record = research.store.get_experiment(experiment_id)
        return {
            "record": record.to_dict(),
            "events": [item.to_dict() for item in research.store.events(experiment_id)],
        }

    @mcp.tool()
    def fugue_experiment_watch(
        experiment_id: str,
        after: int = 0,
        wait_seconds: float = 0.0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Read ordered events after a resumable cursor, with bounded long polling."""

        return watch_experiment_page(
            research,
            experiment_id,
            after=after,
            wait_seconds=wait_seconds,
            limit=limit,
        ).to_dict()

    @mcp.tool()
    def fugue_experiment_cancel(
        experiment_id: str, reason: str, idempotency_key: str
    ) -> dict[str, Any]:
        """Cancel queued or running work without granting any new capability."""

        return research.cancel_experiment(
            experiment_id,
            idempotency_key=idempotency_key,
            reason=reason,
        ).to_dict()

    @mcp.tool()
    def fugue_result_record(
        study_id: str,
        result: dict[str, Any],
        idempotency_key: str,
        expected_revision: int,
        attribution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one evidence-grounded Result to a Study."""

        update = {
            "results": [result],
            "attribution": attribution
            or {"actor_type": "agent", "name": "external-agent"},
        }
        return research.store.update_study(
            study_id,
            study_update_from_dict(update),
            operation_id=idempotency_key,
            expected_revision=expected_revision,
        ).to_dict()

    @mcp.resource("fugue://studies/{study_id}/context")
    def study_context(study_id: str) -> str:
        return json.dumps(research.store.context(study_id).to_dict(), indent=2)

    @mcp.resource("fugue://audits/{audit_id}")
    def trace_audit(audit_id: str) -> str:
        return json.dumps(research.traces.store.get(audit_id).to_dict(), indent=2)

    @mcp.resource("fugue://experiments/{experiment_id}")
    def experiment_status(experiment_id: str) -> str:
        return json.dumps(
            research.store.get_experiment(experiment_id).to_dict(), indent=2
        )

    @mcp.resource("fugue://experiments/{experiment_id}/outcome")
    def experiment_outcome(experiment_id: str) -> str:
        return json.dumps(ExperimentHandle(research, experiment_id).result(), indent=2)

    @mcp.prompt()
    def optimize_agent_use_case(study_id: str) -> str:
        """Use Fugue to turn observed Agent failures into controlled evidence."""

        return (
            _skill_instructions()
            + f"\n\nWork within Study `{study_id}`. Read its context and catalog first."
        )

    @mcp.prompt()
    def design_controlled_experiment(study_id: str, question: str) -> str:
        """Design one controlled comparison using an existing Fugue Study."""

        return (
            _skill_instructions()
            + f"\n\nDesign one experiment for Study `{study_id}` that answers: {question}"
        )

    @mcp.prompt()
    def interpret_experiment(experiment_id: str) -> str:
        """Interpret a terminal experiment without manufacturing a ranking."""

        return (
            _skill_instructions()
            + f"\n\nInterpret experiment `{experiment_id}` from its exact outcome evidence."
        )

    @mcp.prompt()
    def advance_research_cycle(study_id: str, objective: str) -> str:
        """Advance one evidence-to-result-to-next-preview research cycle."""

        return (
            _skill_instructions()
            + "\n\nAdvance exactly one bounded research cycle. Start from the Study's "
            "existing evidence or a registered trace source; do not invent an "
            "observation. If an experiment is already terminal, reconcile and "
            "record its scoped Result before designing anything else. Propose at "
            "most one child experiment, bind it to parent_experiment_ids, "
            "parent_outcome_id, and decision_rationale, and stop after returning "
            "the next eligible preview. Never approve or start that child in the "
            "same cycle. If evidence, policy, or eligibility is insufficient, "
            "record the blocker and stop without a preview."
            + f"\n\nStudy: `{study_id}`\nObjective: {objective}"
        )

    return mcp


def run(repo_root: str | Path, *, env_file: str | Path | None = None) -> None:
    create_mcp_server(repo_root, env_file=env_file).run()


def _skill_instructions() -> str:
    path = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "agent-skills"
        / "optimize-agent-with-fugue"
        / "SKILL.md"
    )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - packaging validation covers this
        raise ResearchError(
            "skill_unavailable",
            "the packaged Fugue optimization skill is unavailable",
            category="evidence",
        ) from exc
    if text.startswith("---\n"):
        _, _, text = text.partition("\n---\n")
    return text.strip()
