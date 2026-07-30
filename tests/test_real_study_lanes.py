from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from fugue.bench import campaign_lifecycle, context
from fugue.bench.ai import get_analysis
from fugue.bench.analysis_contracts import (
    aligned_analysis_declaration_from_dict,
)
from fugue.bench.campaign_lifecycle import (
    CampaignError,
    CampaignService,
    build_experiment_proposal,
    get_campaign,
)
from fugue.bench.context import (
    ContextRuntime,
    PreparedContext,
    RagContextProvider,
    RetrievalQuery,
    get_context_system,
)
from fugue.bench.library import ExecutionLimitsV1, get_experiment
from fugue.bench.operator import ExperimentRequest, OperatorService

REPO_ROOT = Path(__file__).resolve().parents[1]
LOOP_PROJECT = "wandb/fugue-claude-loop-engineering-v1"
HARNESS_PROJECT = "wandb/fugue-harness-experiments-v1"
MEMORY_PROJECT = "wandb/fugue-memory-experiments-v1"
GLM_5_2 = "wandb/zai-org/GLM-5.2"
SONNET_5 = "anthropic/claude-sonnet-5"


def _jobs(experiment_id: str):
    return OperatorService(REPO_ROOT).rendered_jobs(
        ExperimentRequest(experiment_id=experiment_id, preset="canary"),
        run_id=f"preview-{experiment_id}",
        write_configs=False,
    )


def _clean_campaign_service(
    campaign_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> CampaignService:
    source = campaign_lifecycle.resolve_fugue_source_provenance(REPO_ROOT)
    monkeypatch.setattr(
        campaign_lifecycle,
        "resolve_fugue_source_provenance",
        lambda _: {**source, "dirty": False},
    )
    service = CampaignService(REPO_ROOT)
    service.catalog(campaign_id)
    return service


def _preview_campaign(
    *,
    service: CampaignService,
    campaign_id: str,
    experiment_id: str,
    model: str,
    harnesses: tuple[str, ...],
    context_systems: tuple[str, ...],
    variants: tuple[str, ...],
):
    catalog = service.catalog(campaign_id)
    proposal = build_experiment_proposal(
        proposal_id=f"{experiment_id}-canary-001",
        campaign_id=campaign_id,
        catalog_digest=catalog.catalog_digest,
        stage_id="canary",
        research_question="What changes under the registered treatment?",
        hypothesis="The registered treatment may change aligned outcomes.",
        fixed_dimensions=("task", "model", "runtime", "attempt"),
        varied_dimensions=("registered treatment",),
        measured_dimensions=("verifier outcome", "trajectory", "cost", "latency"),
        experiment_id=experiment_id,
        preset_id="canary",
        model=model,
        n_attempts=1,
        n_concurrent=1,
        workloads=("canary",),
        harnesses=harnesses,
        context_systems=context_systems,
        variants=variants,
        n_tasks=2,
        trace_content="full",
    )
    return service.preview(proposal)


def test_claude_loop_lane_is_dedicated_local_eight_plus_eight() -> None:
    experiment = get_experiment("claude-loop-skill-mcp", REPO_ROOT)
    campaign = get_campaign("claude-loop-skill-mcp-v1", REPO_ROOT)
    workloads = {item.id: item for item in experiment.workloads}
    presets = {item.id: item for item in experiment.presets}

    assert experiment.evidence_project == LOOP_PROJECT
    assert experiment.model == SONNET_5
    assert experiment.harnesses == ["claude-code"]
    assert [item.id for item in experiment.variants] == [
        "production",
        "skill-only",
        "mcp-only",
        "combined",
    ]
    assert presets["discovery"].environment == {"type": "docker"}
    assert presets["discovery"].n_tasks == 2
    assert presets["discovery"].n_attempts == 1
    assert (
        presets["discovery"].n_tasks
        * len(workloads["discovery"].variants)
        * len(presets["discovery"].harnesses)
        == 8
    )
    assert presets["holdout"].environment == {"type": "docker"}
    assert presets["holdout"].n_tasks == 4
    assert presets["holdout"].n_attempts == 1
    assert presets["holdout"].selection_lock_required is True
    assert presets["holdout"].selection_lock_kind == "intervention"
    assert presets["holdout"].n_tasks * 2 == 8
    assert [stage.max_cells for stage in campaign.stages] == [8, 8]
    assert campaign.limits.max_total_cells == 16
    assert campaign.require_clean_source is True


def test_harness_lane_resolves_exact_fixed_glm_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = get_experiment("real-harness-study", REPO_ROOT)
    campaign = get_campaign("real-harness-study-v1", REPO_ROOT)
    analysis = get_analysis("real-harness-task-stratified", REPO_ROOT)
    jobs = _jobs(experiment.id)
    service = _clean_campaign_service(campaign.id, monkeypatch)
    plan = _preview_campaign(
        service=service,
        campaign_id=campaign.id,
        experiment_id=experiment.id,
        model=GLM_5_2,
        harnesses=("hermes", "openclaw", "claude-code", "codex"),
        context_systems=("none",),
        variants=("baseline",),
    )

    assert experiment.evidence_project == HARNESS_PROJECT
    assert experiment.evidence_destination is not None
    assert experiment.evidence_destination.project_slug == HARNESS_PROJECT
    assert len(jobs) == plan.cell_count == plan.expected_predictions == 8
    assert {job.task_id for job in jobs} == {
        "sympy__sympy-13031",
        "astropy__astropy-13033",
    }
    assert {job.harness for job in jobs} == {
        "hermes",
        "openclaw",
        "claude-code",
        "codex",
    }
    assert {job.route.provider for job in jobs} == {"wandb"}
    assert {job.route.display_model for job in jobs} == {GLM_5_2}
    assert {job.context_system_id for job in jobs} == {"none"}
    assert {job.config["environment"]["type"] for job in jobs} == {"docker"}
    assert experiment.execution_limits is not None
    assert experiment.execution_limits.wall_time_sec == 900
    limits_digest = experiment.execution_limits.limits_digest
    assert len(limits_digest) == 64
    assert {job.outer_wall_time_sec for job in jobs} == {900}
    assert {job.execution_limits_digest for job in jobs} == {limits_digest}
    assert {
        job.resolved_candidate.execution_definition["execution_limits"][
            "limits_digest"
        ]
        for job in jobs
    } == {limits_digest}
    assert {
        job.config["fugue"]["execution_limits"]["limits_digest"] for job in jobs
    } == {limits_digest}
    assert {
        json.loads(job.env["FUGUE_HARBOR_RESOURCES"])[
            "execution_limits_digest"
        ]
        for job in jobs
    } == {limits_digest}
    assert set(plan.qualification_requirements) == {
        "agent_identity",
        "cost_accounting",
        "route_receipt",
        "runtime_lock",
        "terminal_rows",
    }
    assert campaign.allowed_models == (GLM_5_2,)
    assert campaign.limits.total_cost_usd == 10
    assert campaign.limits.max_total_cells == 8
    assert campaign.require_clean_source is True
    assert campaign.allowed_analyses == ("real-harness-task-stratified",)
    assert analysis.group_by == ("harness", "task_name", "model")
    assert analysis.selection is None
    assert analysis.aligned_analysis is not None
    assert analysis.aligned_analysis.reference_arm == "claude-code"
    assert {arm.id for arm in analysis.aligned_analysis.arms} == {
        "hermes",
        "openclaw",
        "claude-code",
        "codex",
    }
    assert analysis.aligned_analysis.alignment_coordinates == (
        "task_name",
        "trial_index",
    )
    assert "harness" not in analysis.aligned_analysis.alignment_coordinates
    assert {
        contrast.treatment_arms[0]
        for contrast in analysis.aligned_analysis.contrasts
    } == {"hermes", "openclaw", "codex"}
    assert {
        dimension.role
        for contrast in analysis.aligned_analysis.contrasts
        for dimension in contrast.dimensions
        if dimension.id == "outer-wall-timeout"
    } == {"infrastructure"}
    assert experiment.research_view is not None
    assert experiment.research_view.pass_rule.startswith(
        "Report Harbor verifier success"
    )


def test_memory_lane_resolves_exact_locked_sonnet_factorial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = get_experiment("real-memory-study", REPO_ROOT)
    campaign = get_campaign("real-memory-study-v1", REPO_ROOT)
    analysis = get_analysis("real-memory-factorial", REPO_ROOT)
    jobs = _jobs(experiment.id)
    service = _clean_campaign_service(campaign.id, monkeypatch)
    plan = _preview_campaign(
        service=service,
        campaign_id=campaign.id,
        experiment_id=experiment.id,
        model=SONNET_5,
        harnesses=("claude-code",),
        context_systems=("none", "rag-dense"),
        variants=("baseline", "rag-dense", "policy-only", "combined"),
    )

    assert experiment.evidence_project == MEMORY_PROJECT
    assert experiment.evidence_destination is not None
    assert experiment.evidence_destination.project_slug == MEMORY_PROJECT
    assert len(jobs) == plan.cell_count == plan.expected_predictions == 8
    assert {job.task_id for job in jobs} == {
        "sympy__sympy-18199",
        "sphinx-doc__sphinx-9461",
    }
    assert {job.harness for job in jobs} == {"claude-code"}
    assert {job.route.provider for job in jobs} == {"anthropic"}
    assert {job.route.display_model for job in jobs} == {SONNET_5}
    assert {job.variant_id for job in jobs} == {
        "baseline",
        "rag-dense",
        "policy-only",
        "combined",
    }
    assert {
        job.variant_id: job.context_system_id for job in jobs
    } == {
        "baseline": "none",
        "rag-dense": "rag-dense",
        "policy-only": "none",
        "combined": "rag-dense",
    }
    assert {job.config["environment"]["type"] for job in jobs} == {"docker"}
    assert set(plan.qualification_requirements) == {
        "agent_identity",
        "cost_accounting",
        "route_receipt",
        "runtime_lock",
        "terminal_rows",
    }
    assert "context_system:rag-dense" in plan.component_digests
    assert campaign.allowed_models == (SONNET_5,)
    assert campaign.limits.total_cost_usd == 10
    assert campaign.limits.max_total_cells == 8
    assert campaign.require_clean_source is True
    assert campaign.allowed_analyses == ("real-memory-factorial",)
    assert analysis.group_by == (
        "variant_id",
        "task_name",
        "context_system_id",
        "model",
    )
    assert analysis.selection is None
    assert analysis.aligned_analysis is not None
    assert analysis.aligned_analysis.reference_arm == "baseline"
    assert {
        contrast.treatment_arms[0]
        for contrast in analysis.aligned_analysis.contrasts
    } == {"rag-dense", "policy-only", "combined"}
    [interaction] = analysis.aligned_analysis.interactions
    assert interaction.cell_arms == {
        "00": "baseline",
        "10": "rag-dense",
        "01": "policy-only",
        "11": "combined",
    }
    assert tuple(item.id for item in interaction.factors) == (
        "retrieval",
        "evidence-policy",
    )
    assert experiment.research_view is not None
    assert experiment.research_view.arm_factor_levels == {
        "baseline": {"retrieval": "off", "evidence-policy": "standard"},
        "rag-dense": {"retrieval": "dense", "evidence-policy": "standard"},
        "policy-only": {"retrieval": "off", "evidence-policy": "required"},
        "combined": {"retrieval": "dense", "evidence-policy": "required"},
    }


def test_harness_outer_wall_changes_execution_not_behavior_identity() -> None:
    experiment = get_experiment("real-harness-study", REPO_ROOT)
    service = OperatorService(REPO_ROOT)
    request = ExperimentRequest(
        experiment_id=experiment.id,
        preset="canary",
        harnesses=("claude-code",),
        n_tasks=1,
    )
    [original] = service.rendered_jobs(
        request,
        run_id="preview-harness-limit-original",
        write_configs=False,
        experiment=experiment,
    )
    changed = replace(
        experiment,
        execution_limits=ExecutionLimitsV1(wall_time_sec=901),
    )
    [revised] = service.rendered_jobs(
        request,
        run_id="preview-harness-limit-revised",
        write_configs=False,
        experiment=changed,
    )

    assert original.candidate_id == revised.candidate_id
    assert (
        original.resolved_candidate.execution_fingerprint
        != revised.resolved_candidate.execution_fingerprint
    )


def test_real_analysis_declarations_round_trip_and_reject_tampering() -> None:
    declaration = get_analysis(
        "real-memory-factorial", REPO_ROOT
    ).aligned_analysis
    assert declaration is not None
    payload = declaration.to_dict()

    assert (
        aligned_analysis_declaration_from_dict(payload).declaration_digest
        == declaration.declaration_digest
    )
    with pytest.raises(
        ValueError, match="aligned analysis declaration digest does not match"
    ):
        aligned_analysis_declaration_from_dict(
            {**payload, "declaration_digest": "0" * 64}
        )
    with pytest.raises(ValueError, match="unknown aligned analysis declaration"):
        aligned_analysis_declaration_from_dict({**payload, "winner": "combined"})


def test_rag_dense_failure_cannot_fall_back_to_bm25(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "prepared" / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "chunks.jsonl").write_text(
        json.dumps(
            {
                "id": "src/app.py:1",
                "path": "src/app.py",
                "start_line": 1,
                "end_line": 1,
                "text": "target behavior",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prepared = PreparedContext(
        system_id="rag-dense",
        cache_key="a" * 64,
        path=artifact.parent,
        manifest={},
        metrics={},
    )
    runtime = ContextRuntime(REPO_ROOT, tmp_path / "cache", {})
    spec = get_context_system("rag-dense", REPO_ROOT)
    provider = RagContextProvider()
    lexical_used = False

    def fake_bm25(query, chunks):
        nonlocal lexical_used
        del query
        lexical_used = True
        return [{**chunks[0], "score": 1.0}]

    def fail_dense(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("dense artifact unavailable")

    monkeypatch.setattr(context, "_bm25", fake_bm25)
    monkeypatch.setattr(context, "_dense_search", fail_dense)

    with pytest.raises(RuntimeError, match="dense artifact unavailable"):
        asyncio.run(
            provider.retrieve(
                spec,
                RetrievalQuery(id="query", text="target behavior", top_k=1),
                prepared,
                runtime,
            )
        )
    assert lexical_used is False


@pytest.mark.parametrize(
    (
        "campaign_id",
        "experiment_id",
        "model",
        "harnesses",
        "context_systems",
        "variants",
    ),
    [
        (
            "real-harness-study-v1",
            "real-harness-study",
            SONNET_5,
            ("hermes", "openclaw", "claude-code", "codex"),
            ("none",),
            ("baseline",),
        ),
        (
            "real-memory-study-v1",
            "real-memory-study",
            GLM_5_2,
            ("claude-code",),
            ("none", "rag-dense"),
            ("baseline", "rag-dense", "policy-only", "combined"),
        ),
    ],
)
def test_real_lane_campaigns_reject_model_route_drift(
    campaign_id: str,
    experiment_id: str,
    model: str,
    harnesses: tuple[str, ...],
    context_systems: tuple[str, ...],
    variants: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _clean_campaign_service(campaign_id, monkeypatch)

    with pytest.raises(CampaignError, match="does not allow model"):
        _preview_campaign(
            service=service,
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            model=model,
            harnesses=harnesses,
            context_systems=context_systems,
            variants=variants,
        )


def test_real_lane_assets_never_route_to_legacy_news_project() -> None:
    paths = (
        REPO_ROOT / "configs/fugue/experiments/claude-loop-skill-mcp.yaml",
        REPO_ROOT / "configs/fugue/experiments/real-harness-study.yaml",
        REPO_ROOT / "configs/fugue/experiments/real-memory-study.yaml",
        REPO_ROOT / "configs/fugue/campaigns/claude-loop-skill-mcp-v1.yaml",
        REPO_ROOT / "configs/fugue/campaigns/real-harness-study-v1.yaml",
        REPO_ROOT / "configs/fugue/campaigns/real-memory-study-v1.yaml",
        REPO_ROOT / "datasets/real-study-lanes/harness-canary-v1.yaml",
        REPO_ROOT / "datasets/real-study-lanes/memory-canary-v1.yaml",
        REPO_ROOT / "docs/demos/real-study-lanes.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "news" + "-research-agent" not in text
    assert LOOP_PROJECT in text
    assert HARNESS_PROJECT in text
    assert MEMORY_PROJECT in text
