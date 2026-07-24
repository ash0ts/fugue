from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fugue.bench.campaigns import get_campaign
from fugue.bench.export import _evidence_use_rewards
from fugue.bench.library import get_experiment
from fugue.bench.manifest import load_manifest
from fugue.bench.operator import ExperimentRequest, OperatorService
from fugue.research.agent_contracts import (
    build_trace_audit_draft,
    build_trace_selection,
)
from fugue.research.service import ResearchService
from fugue.research.store import StudyStore
from fugue.research.task_recipes import (
    reviewed_task_recipe_ids,
    task_recipe_draft_from_dict,
)
from fugue.research.traces import TraceSourceRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets/enterprise-evidence-use-v1"


def _run_task_verifier(
    tmp_path: Path,
    task_id: str,
    artifact: dict[str, str],
) -> tuple[int, dict[str, float]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact_path = tmp_path / f"{task_id}-artifact.json"
    reward_path = tmp_path / f"{task_id}-reward.json"
    artifact_path.write_text(json.dumps(artifact))
    script = (DATASET_ROOT / task_id / "tests/test.sh").read_text()
    python_source = script.split("python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    python_source = python_source.replace(
        'Path("/logs/artifacts/research-brief.json")',
        f"Path({str(artifact_path)!r})",
    ).replace(
        'Path("/logs/verifier/reward.json")',
        f"Path({str(reward_path)!r})",
    )
    exit_code = 0
    try:
        exec(compile(python_source, f"{task_id}/tests/test.sh", "exec"), {})
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    return exit_code, json.loads(reward_path.read_text())


def _service(tmp_path: Path) -> ResearchService:
    def fetch(payload: dict[str, Any]) -> list[dict[str, Any]]:
        if "call_ids" in payload["filter"]:
            return [
                {
                    "id": call_id,
                    "trace_id": f"trace-{index}",
                    "op_name": "open_document",
                }
                for index, call_id in enumerate(payload["filter"]["call_ids"], 1)
            ]
        return [
            {
                "id": f"root-{index}",
                "trace_id": trace_id,
                "op_name": "answer_enterprise_question",
                "started_at": f"2026-07-24T12:0{index}:00Z",
                "summary": {"weave": {"status": "success"}},
                "attributes": {
                    "demo.dataset": "enterprise-evidence-agent-v1",
                    "demo.outcome": "evidence-not-used",
                    "demo.needs_review": True,
                    "demo.synthetic": True,
                },
            }
            for index, trace_id in enumerate(payload["filter"]["trace_ids"], 1)
        ]

    registry = TraceSourceRegistry.from_mapping(
        {
            "version": 1,
            "sources": [
                {
                    "id": "enterprise-evidence-agent",
                    "adapter": "weave",
                    "allowed_projects": ["demo/enterprise-evidence"],
                    "allowed_fields": ["status", "operation"],
                    "allowed_filters": ["status"],
                }
            ],
        },
        root=tmp_path,
        weave_fetchers={"enterprise-evidence-agent": fetch},
    )
    service = ResearchService(
        REPO_ROOT,
        store=StudyStore(tmp_path),
        trace_registry=registry,
    )
    service.store.create_study(
        study_id="enterprise-evidence-study",
        title="Enterprise evidence use",
        campaign_id="enterprise-evidence-use-v1",
        question="Does search help if the Agent does not inspect its source?",
        operation_id="create-enterprise-evidence-study",
    )
    return service


def test_enterprise_evidence_previews_exact_canary_and_primary() -> None:
    operator = OperatorService(REPO_ROOT)
    canary = operator.preview(
        ExperimentRequest(
            experiment_id="enterprise-evidence-use-v1",
            preset="canary",
        )
    )
    primary = operator.preview(
        ExperimentRequest(
            experiment_id="enterprise-evidence-use-v1",
            preset="primary",
        )
    )

    assert canary.cells == canary.estimated_trials == 8
    assert primary.cells == primary.estimated_trials == 64
    assert canary.harnesses == primary.harnesses == ("claude-code", "codex")
    assert (
        canary.variants
        == primary.variants
        == (
            "baseline",
            "inspect-only",
            "search-and-inspect",
            "search-only",
        )
    )
    experiment = get_experiment("enterprise-evidence-use-v1", REPO_ROOT)
    assert {preset.id: preset.n_concurrent for preset in experiment.presets} == {
        "canary": 1,
        "primary": 1,
    }
    assert len({cell.task_id for cell in primary.matrix_cells}) == 4


def test_enterprise_recipe_uses_fresh_versioned_demo_campaign() -> None:
    campaign = get_campaign("enterprise-evidence-use-demo-v3", REPO_ROOT)
    stages = {stage.id: stage for stage in campaign.stages}

    assert campaign.allowed_experiments == ("enterprise-evidence-use-v1",)
    assert campaign.revision == "v3"
    assert stages["canary"].max_proposals == 1
    assert stages["primary"].max_proposals == 1
    assert campaign.limits.max_total_cells == 72
    assert campaign.limits.max_concurrent == 1


def test_enterprise_recipe_requires_exact_reviewed_four_call_cohort(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    selection = build_trace_selection(
        project="demo/enterprise-evidence",
        mode="selected",
        call_ids=[f"reviewed-{index}" for index in range(1, 5)],
        filters={},
        max_traces=4,
    )
    audit = service.traces.run(
        service.traces.preview(
            "enterprise-evidence-study",
            build_trace_audit_draft(
                study_id="enterprise-evidence-study",
                source_id="enterprise-evidence-agent",
                objective="Inspect failures where the current source was not used.",
                fields=["status", "operation"],
                filters={},
                max_traces=4,
                selection=selection.to_dict(),
            ),
        ),
        operation_id="audit-enterprise-evidence",
    )
    recipe = service.task_recipes.derive_preview(
        "enterprise-evidence-study",
        task_recipe_draft_from_dict(
            {
                "schema_version": 1,
                "study_id": "enterprise-evidence-study",
                "audit_id": audit.id,
                "recipe_id": "enterprise-evidence-use-v1",
                "objective": "Separate search availability from source inspection.",
            },
            require_digest=False,
        ),
    )

    assert recipe.eligible is True
    assert recipe.provenance["source_dataset"] == "enterprise-evidence-agent-v1"
    assert recipe.provenance["needs_review_root_count"] == 4
    assert (
        recipe.experiment_binding["campaign_id"]
        == "enterprise-evidence-use-demo-v3"
    )
    assert recipe.experiment_binding["estimated_cells"] == 8
    assert recipe.sanitization_report["copied_trace_content"] is False
    assert reviewed_task_recipe_ids() == (
        "enterprise-evidence-use-v1",
        "support-data-authority-v1",
    )


def test_enterprise_tasks_expose_schema_but_keep_expected_facts_in_verifiers() -> None:
    manifest = load_manifest(REPO_ROOT / "datasets/enterprise-evidence-use-v1.yaml")
    assert len(manifest.tasks) == 4
    for task in manifest.tasks:
        root = DATASET_ROOT / task.id
        instruction = (root / "instruction.md").read_text()
        verifier = (root / "tests/test.sh").read_text()
        assert "research-brief.json" in instruction
        assert "source_document" in instruction
        assert "answer_facts_correct" not in instruction
        assert "answer_facts_correct" in verifier
        assert "current_document_cited" in verifier
        assert "unsupported_claims_absent" in verifier
        documents = list((root / "environment/documents").glob("*.md"))
        assert any("superseded" in path.name for path in documents)
        assert any("draft" in path.name for path in documents)


def test_evidence_use_reward_contract_is_all_or_nothing() -> None:
    values = _evidence_use_rewards(
        {
            "rewards": {
                "artifact_schema_valid": 1.0,
                "answer_facts_correct": 1.0,
                "current_document_cited": 1.0,
                "current_document_used": 1.0,
                "unsupported_claims_absent": 1.0,
            }
        }
    )

    assert values == {
        "artifact_schema_valid": 1.0,
        "answer_facts_correct": 1.0,
        "current_document_cited": 1.0,
        "current_document_used": 1.0,
        "unsupported_claims_absent": 1.0,
    }


def test_expense_verifier_accepts_semantically_equivalent_agent_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = (
        {
            "question_id": "expense-policy-limit",
            "answer": "USD 125 per attendee, including tax and gratuity",
            "source_document": "documents/expense-policy-rev-7.md",
            "source_revision": "revision 7",
            "brief": (
                "Revision 7 is current. Revision 6 listed USD 90 and revision 8 "
                "drafted USD 150, but neither is in effect."
            ),
        },
        {
            "question_id": "expense-policy-limit",
            "answer": "$125 per attendee (including tax and gratuity)",
            "source_document": "expense-policy-rev-7.md",
            "source_revision": "7",
            "brief": "The current source is revision 7.",
        },
    )

    for index, artifact in enumerate(artifacts):
        exit_code, rewards = _run_task_verifier(
            tmp_path / str(index),
            "expense-policy-limit",
            artifact,
        )
        assert exit_code == 0
        assert rewards == {
            "answer_facts_correct": 1.0,
            "artifact_schema_valid": 1.0,
            "current_document_cited": 1.0,
            "current_document_used": 1.0,
            "reward": 1.0,
            "unsupported_claims_absent": 1.0,
        }


def test_expense_verifier_rejects_a_superseded_answer(tmp_path: Path) -> None:
    exit_code, rewards = _run_task_verifier(
        tmp_path,
        "expense-policy-limit",
        {
            "question_id": "expense-policy-limit",
            "answer": "USD 90 per attendee",
            "source_document": "documents/expense-policy-rev-6-superseded.md",
            "source_revision": "revision 6",
            "brief": "Revision 6 is superseded.",
        },
    )

    assert exit_code == 1
    assert rewards["answer_facts_correct"] == 0.0
    assert rewards["current_document_cited"] == 0.0
    assert rewards["reward"] == 0.0


def test_primary_task_verifiers_accept_human_formatting_variation(
    tmp_path: Path,
) -> None:
    artifacts = {
        "vendor-retention": {
            "question_id": "vendor-retention",
            "answer": "Active data: 30 days. Backups: 90 days.",
            "source_document": "vendor-retention-2026-04.md",
            "source_revision": "April 2026",
            "brief": "The April 2026 agreement is authoritative.",
        },
        "equipment-allowance": {
            "question_id": "equipment-allowance",
            "answer": "US: 1,800 USD annually. Japan: 240,000 JPY annually.",
            "source_document": "equipment-allowance-2026.md",
            "source_revision": "2026 policy",
            "brief": "Japan uses the regional exception.",
        },
        "incident-escalation": {
            "question_id": "incident-escalation",
            "answer": (
                "Page Security and the Executive Incident Lead within ten minutes."
            ),
            "source_document": "incident-escalation-v5.md",
            "source_revision": "revision 5",
            "brief": "Version 5 is authoritative.",
        },
    }

    for task_id, artifact in artifacts.items():
        exit_code, rewards = _run_task_verifier(tmp_path / task_id, task_id, artifact)
        assert exit_code == 0
        assert rewards["reward"] == 1.0


def test_enterprise_experiment_declares_factorial_research_view() -> None:
    view = get_experiment("enterprise-evidence-use-v1", REPO_ROOT).research_view

    assert view is not None
    assert view.arm_factor_levels["search-and-inspect"] == {
        "repository-search": "on",
        "source-inspection": "required",
    }
    assert [stage.id for stage in view.mechanism_stages] == [
        "search-available",
        "search-invoked",
        "current-source-returned",
        "current-source-opened",
        "current-source-cited",
        "current-source-used",
    ]
    assert view.scorers[0].kind == "deterministic"
    assert all(scorer.kind != "llm_judge" for scorer in view.scorers)
