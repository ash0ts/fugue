from __future__ import annotations

from pathlib import Path

from fugue.bench.library import get_experiment
from fugue.bench.operator import ExperimentRequest, OperatorService

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_MODELS = (
    "wandb/zai-org/GLM-5.2",
    "wandb/moonshotai/Kimi-K2.7-Code",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4-8",
)
CEILING_MODEL = "anthropic/claude-fable-5"


def _jobs(model: str, preset: str):
    return OperatorService(REPO_ROOT).rendered_jobs(
        ExperimentRequest(
            experiment_id="swe-frontier-harness",
            preset=preset,
            model=model,
        ),
        run_id=f"preview-{preset}-{model.replace('/', '-')}",
        write_configs=False,
    )


def test_frontier_campaign_previews_exact_cohorts() -> None:
    experiment = get_experiment("swe-frontier-harness", REPO_ROOT)

    assert experiment.judge_model is None
    for model in MAIN_MODELS:
        jobs = _jobs(model, "discovery")
        assert len(jobs) == 32
        assert {job.route.display_model for job in jobs} == {model}
        assert {job.harness for job in jobs} == {
            "hermes",
            "openclaw",
            "claude-code",
            "codex",
        }
        assert len({job.task_id for job in jobs}) == 8
        assert {job.context_system_id for job in jobs} == {"none"}
        assert all(job.evaluation_rubrics == () for job in jobs)

    ceiling = _jobs(CEILING_MODEL, "frontier-ceiling")
    assert len(ceiling) == 16
    assert {job.harness for job in ceiling} == {"claude-code", "codex"}
    assert len({job.task_id for job in ceiling}) == 8


def test_frontier_cohorts_share_examples_but_not_candidates() -> None:
    glm = _jobs(MAIN_MODELS[0], "discovery")
    kimi = _jobs(MAIN_MODELS[1], "discovery")
    examples = {
        (job.harness, job.task_id): job.comparison_example_id for job in glm
    }

    assert examples == {
        (job.harness, job.task_id): job.comparison_example_id for job in kimi
    }
    glm_candidates = {(job.harness, job.task_id): job.candidate_id for job in glm}
    kimi_candidates = {(job.harness, job.task_id): job.candidate_id for job in kimi}
    assert all(
        glm_candidates[key] != kimi_candidates[key] for key in glm_candidates
    )
    assert all(
        "scorer" not in job.resolved_candidate.definition
        and "judge" not in job.resolved_candidate.definition
        for job in (*glm, *kimi)
    )


def test_frontier_canaries_are_one_task_and_side_effect_free() -> None:
    main = OperatorService(REPO_ROOT).preview(
        ExperimentRequest(
            experiment_id="swe-frontier-harness",
            preset="canary",
            model=MAIN_MODELS[0],
        )
    )
    ceiling = OperatorService(REPO_ROOT).preview(
        ExperimentRequest(
            experiment_id="swe-frontier-harness",
            preset="frontier-ceiling-canary",
            model=CEILING_MODEL,
        )
    )

    assert (main.cells, main.applicable_cells, main.estimated_trials) == (4, 4, 4)
    assert (ceiling.cells, ceiling.applicable_cells, ceiling.estimated_trials) == (
        2,
        2,
        2,
    )
