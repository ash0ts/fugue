from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import fugue.bench.operator as operator_module
from fugue.bench.ai import AssetDraft
from fugue.bench.candidates import ResolvedCandidate, stable_digest
from fugue.bench.execution import (
    CellOutcome,
    plan_cells,
    read_run_manifest,
    write_run_manifest,
)
from fugue.bench.export import PublicationResult, PublishedEvaluation
from fugue.bench.intervention_provenance import (
    build_intervention_component_lock,
)
from fugue.bench.library import (
    ContextSelection,
    EvaluationGenerationSpec,
    IntegrationSelection,
    RubricScorerSelection,
    WorkloadSpec,
)
from fugue.bench.operator import ExperimentRequest, OperatorService, as_json
from fugue.bench.reproducibility import (
    RunSnapshotV1,
    build_evaluation_asset_lock,
    build_run_snapshot,
    read_evaluation_asset_lock,
    verify_snapshot,
    write_evaluation_asset_lock,
)
from fugue.bench.scoring import (
    build_intervention_selection_lock,
    write_intervention_selection_lock,
)
from fugue.bench.services import GRAPHITI_SERVICE, ManagedServiceStatus
from fugue.model_plane import (
    EvidenceDestinationV1,
    model_route_identity,
    trace_project_slug,
)
from fugue.preflight import PreflightCheck


def make_operator_repo(tmp_path: Path) -> OperatorService:
    (tmp_path / "configs/fugue/experiments").mkdir(parents=True)
    (tmp_path / "configs/fugue/context-systems").mkdir(parents=True)
    (tmp_path / "configs/fugue/prompts").mkdir(parents=True)
    (tmp_path / "configs/fugue/skills/demo-skill").mkdir(parents=True)
    (tmp_path / "configs/fugue/agent-presets").mkdir(parents=True)
    (tmp_path / "datasets").mkdir()
    (tmp_path / "configs/fugue/context-systems/none.yaml").write_text(
        """
id: none
title: No added context
description: Control
provider: fugue.bench.context:EmptyContextProvider
version: "1"
capabilities: [prepare, retrieve, bind, ingest, sequence, serve]
deliveries: [portable]
serve_deliveries: [portable]
license: Fugue
"""
    )
    (tmp_path / "datasets/demo.yaml").write_text(
        """
dataset: {ref: demo/tasks, version: v1}
model: openai/gpt-5
k: 1
n_concurrent: 1
jobs_dir: jobs/demo
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-one, repository: {type: git, url: https://github.com/test/repo, commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}}
"""
    )
    (tmp_path / "configs/fugue/experiments/demo.yaml").write_text(
        """
id: demo
title: Demo
manifest: datasets/demo.yaml
model: openai/gpt-5
harnesses: [codex]
variants:
  - {id: baseline, label: Baseline, context: {system_id: none, delivery: portable}}
n_attempts: 1
n_concurrent: 1
jobs_dir: jobs/demo
trace_content: full
"""
    )
    (tmp_path / "configs/fugue/prompts/demo-prompt.md").write_text(
        "# Demo prompt\n\nInspect the repository before editing.\n"
    )
    (tmp_path / "configs/fugue/skills/demo-skill/SKILL.md").write_text(
        "# Demo skill\n\nUse focused repository search.\n"
    )
    (tmp_path / "configs/fugue/agent-presets/demo-maintainer.yaml").write_text(
        """
id: demo-maintainer
title: Demo maintainer
role: maintainer
base_experiment_id: demo
candidate:
  harness: codex
  model: openai/gpt-5
  prompt_id: demo-prompt
  skills: [demo-skill]
  context: {system_id: none, delivery: portable}
evidence:
  suite_id: demo-v1
  suite_digest: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  base_commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  run_ids: [run-1]
  analysis_snapshot: snapshot-1
  metrics: {pass_rate: 1.0}
"""
    )
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=model-secret\n"
        "WANDB_API_KEY=trace-secret\n"
        "WANDB_ENTITY=team\n"
        "WANDB_PROJECT=fugue-experiments\n"
    )
    return OperatorService(tmp_path, tmp_path / ".env")


def test_intervention_lock_freezes_two_arm_holdout_and_candidate_identity(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    (tmp_path / "configs/fugue/experiments/loop.yaml").write_text(
        """
id: loop
title: Skill and MCP loop
manifest: datasets/demo.yaml
model: openai/gpt-5
tags: [task-suite:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee]
harnesses: [codex]
default_preset: discovery
presets:
  discovery: {n_tasks: 1, n_attempts: 1}
  holdout:
    n_tasks: 1
    n_attempts: 1
    selection_lock_required: true
    selection_lock_kind: intervention
variants:
  - {id: production, label: Production, context: {system_id: none, delivery: portable}}
  - {id: skill-only, label: Skill only, skills: [demo-skill], context: {system_id: none, delivery: portable}}
  - {id: mcp-only, label: MCP only, prompt_id: demo-prompt, context: {system_id: none, delivery: portable}}
  - {id: combined, label: Combined, prompt_id: demo-prompt, skills: [demo-skill], context: {system_id: none, delivery: portable}}
"""
    )
    (tmp_path / ".gitignore").write_text(".fugue/\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fugue Tests",
            "-c",
            "user.email=fugue-tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    discovery = service.rendered_jobs(
        ExperimentRequest(experiment_id="loop", preset="discovery"),
        run_id="discovery",
        write_configs=False,
    )
    candidate_by_variant = {job.variant_id: job.candidate_id for job in discovery}
    lock = build_intervention_selection_lock(
        experiment_id="loop",
        source_commit=source_commit,
        source_tree=source_tree,
        source_dirty_digest="",
        failure_lock_sha256="f" * 64,
        discovery_suite_sha256="d" * 64,
        holdout_suite_sha256="e" * 64,
        analysis_snapshot_sha256="a" * 64,
        discovery_run_snapshot_sha256s=("b" * 64,),
        comparison_example_ids=("c" * 64,),
        discovery_variant_ids=(
            "production",
            "skill-only",
            "mcp-only",
            "combined",
        ),
        baseline_variant_id="production",
        selected_variant_id="combined",
        selected_components=(
            build_intervention_component_lock(
                kind="skill",
                component_id="loop-intervention-skill",
                lock_digest="7" * 64,
                repository="https://github.com/wandb/fugue",
                source_commit="8" * 40,
                source_tree="9" * 40,
            ),
            build_intervention_component_lock(
                kind="mcp",
                component_id="loop-intervention-mcp",
                lock_digest="a" * 64,
                repository="https://github.com/wandb/wandb-mcp-server",
                source_commit="b" * 40,
                source_tree="c" * 40,
                release_target="wandb-mcp-server Python package 0.4",
                superseded_release_candidate_sha="d" * 40,
                release_requalification_required=True,
            ),
        ),
        rankings=tuple(
            {
                "variant_id": variant_id,
                "candidate_digest": candidate_by_variant[variant_id],
            }
            for variant_id in ("production", "skill-only", "mcp-only", "combined")
        ),
        decision="recommend",
        rationale="the combined arm met the preregistered discovery gate",
        failure_locked_at="2026-07-30T10:00:00Z",
        suites_frozen_at="2026-07-30T10:05:00Z",
        discovery_completed_at="2026-07-30T11:00:00Z",
        selection_locked_at="2026-07-30T11:05:00Z",
    )
    lock_path = write_intervention_selection_lock(
        tmp_path / ".fugue/selection.json", lock
    )

    holdout = service.rendered_jobs(
        ExperimentRequest(
            experiment_id="loop",
            preset="holdout",
            selection_lock=lock_path,
        ),
        run_id="holdout",
        write_configs=False,
    )
    assert {job.variant_id for job in holdout} == {"production", "combined"}

    drifted_suite = replace(
        service.experiment("loop"),
        tags=["task-suite:" + "0" * 64],
    )
    with pytest.raises(ValueError, match="holdout Task Suite differs"):
        service.rendered_jobs(
            ExperimentRequest(
                experiment_id="loop",
                preset="holdout",
                selection_lock=lock_path,
            ),
            run_id="holdout-drifted-suite",
            write_configs=False,
            experiment=drifted_suite,
        )

    (tmp_path / "configs/fugue/skills/demo-skill/SKILL.md").write_text(
        "# Demo skill\n\nThis drift occurred after discovery.\n"
    )
    with pytest.raises(ValueError, match="source tree differs"):
        service.rendered_jobs(
            ExperimentRequest(
                experiment_id="loop",
                preset="holdout",
                selection_lock=lock_path,
            ),
            run_id="holdout-drifted",
            write_configs=False,
        )


def test_managed_service_lifecycle_selects_only_requested_context_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    request = ExperimentRequest(experiment_id="demo", systems=("none", "graphiti"))
    calls: list[tuple[str, tuple[object, ...]]] = []
    status = ManagedServiceStatus(
        GRAPHITI_SERVICE.id,
        "healthy",
        True,
        "container is healthy",
        GRAPHITI_SERVICE.container_name,
        GRAPHITI_SERVICE.image,
        GRAPHITI_SERVICE.host_uri,
    )

    monkeypatch.setattr(
        "fugue.bench.operator.managed_service_statuses",
        lambda specs, **kwargs: calls.append(("status", tuple(specs))) or (status,),
    )
    monkeypatch.setattr(
        "fugue.bench.operator.start_managed_services",
        lambda specs, **kwargs: calls.append(("start", tuple(specs))) or (status,),
    )
    monkeypatch.setattr(
        "fugue.bench.operator.stop_managed_services",
        lambda specs, **kwargs: calls.append(("stop", tuple(specs))) or (status,),
    )

    assert service.service_status(request) == (status,)
    assert service.start_services(request) == (status,)
    assert service.stop_services(request) == (status,)
    assert calls == [
        ("status", (GRAPHITI_SERVICE,)),
        ("start", (GRAPHITI_SERVICE,)),
        ("stop", (GRAPHITI_SERVICE,)),
    ]


def test_external_graphiti_uri_does_not_require_managed_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    (tmp_path / "configs/fugue/context-systems/graphiti.yaml").write_text(
        """
id: graphiti
title: Graphiti
provider: fugue.bench.context:EmptyContextProvider
version: test
capabilities: [prepare, retrieve, bind]
deliveries: [portable]
support: experimental
required_env: [FUGUE_GRAPHITI_URI, FUGUE_GRAPHITI_USER, FUGUE_GRAPHITI_PASSWORD]
"""
    )
    with (tmp_path / ".env").open("a") as handle:
        handle.write(
            "FUGUE_GRAPHITI_URI=bolt+s://graph.example.test\n"
            "FUGUE_GRAPHITI_USER=neo4j\n"
            "FUGUE_GRAPHITI_PASSWORD=external-password\n"
        )
    observed: list[tuple[object, ...]] = []
    monkeypatch.setattr("fugue.preflight.run_preflight", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "fugue.bench.operator.managed_service_statuses",
        lambda specs, **kwargs: observed.append(tuple(specs)) or (),
    )

    checks = service.preflight(
        ExperimentRequest(experiment_id="demo", systems=("graphiti",)),
        live=False,
    )

    assert observed == [()]
    assert not any(check.name.startswith("managed service") for check in checks)
    assert all(
        check.ok for check in checks if check.name.startswith("context graphiti: env:")
    )


def test_operator_preflight_and_prepare_rebind_declared_destination_a_b_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_operator_repo(tmp_path)
    base = service.experiment("demo")

    def lane(project: str):
        destination = EvidenceDestinationV1(
            entity="team",
            project=project,
            api_base_url="https://api.wandb.ai",
            trace_base_url="https://trace.wandb.ai",
            app_base_url="https://wandb.ai",
        )
        return replace(
            base,
            evidence_project=destination.project_slug,
            evidence_destination=destination,
        )

    lane_a = lane("lane-a")
    lane_b = lane("lane-b")
    preflight_projects: list[str] = []
    preparation_projects: list[str] = []

    def capture_preflight(*_args, **kwargs):
        preflight_projects.append(trace_project_slug(kwargs["env"]))
        return []

    async def capture_preparation(
        _name,
        _metadata,
        env,
        operation,
        _summarize,
    ):
        preparation_projects.append(trace_project_slug(env))
        return await operation()

    monkeypatch.setattr("fugue.preflight.run_preflight", capture_preflight)
    monkeypatch.setattr(
        operator_module,
        "trace_async_operation",
        capture_preparation,
    )

    request = ExperimentRequest(experiment_id="demo")
    for experiment in (lane_a, lane_b, lane_a):
        service.preflight(request, live=False, experiment=experiment)
        service.prepare_context(request, experiment=experiment)

    assert preflight_projects == [
        "team/lane-a",
        "team/lane-b",
        "team/lane-a",
    ]
    assert preparation_projects == [
        "team/lane-a",
        "team/lane-b",
        "team/lane-a",
    ]


def test_operator_full_prepare_rebinds_declared_destination_a_b_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_operator_repo(tmp_path)
    base = service.experiment("demo")

    def lane(project: str):
        destination = EvidenceDestinationV1(
            entity="team",
            project=project,
            api_base_url="https://api.wandb.ai",
            trace_base_url="https://trace.wandb.ai",
            app_base_url="https://wandb.ai",
        )
        return replace(
            base,
            evidence_project=destination.project_slug,
            evidence_destination=destination,
        )

    preparation_projects: list[str] = []

    def capture_runtime(**kwargs):
        preparation_projects.append(trace_project_slug(kwargs["env"]))
        return SimpleNamespace(env=kwargs["env"])

    monkeypatch.setattr(operator_module, "ContextRuntime", capture_runtime)
    monkeypatch.setattr(
        operator_module,
        "materialize_manifest_dataset",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(operator_module, "agent_runtime_spec", lambda _name: None)
    monkeypatch.setattr(service, "prepare_context", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        operator_module,
        "prepare_task_runtime",
        lambda *args, **kwargs: {
            "image": "fugue-task:test",
            "image_id": "sha256:" + "a" * 64,
            "recipe_sha256": "b" * 64,
        },
    )

    request = ExperimentRequest(experiment_id="demo")
    for experiment in (lane("lane-a"), lane("lane-b"), lane("lane-a")):
        service.prepare(request, experiment=experiment)

    assert preparation_projects == [
        "team/lane-a",
        "team/lane-b",
        "team/lane-a",
    ]


def test_operator_status_masks_secrets_and_links_to_agents(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    status = service.status(ExperimentRequest(experiment_id="demo"))
    payload = as_json(status)

    assert status.model_key_present is True
    assert status.trace_key_present is True
    assert status.links.agents == (
        "https://wandb.ai/team/fugue-experiments/weave/agents"
    )
    assert "model-secret" not in payload
    assert "trace-secret" not in payload
    assert "catalog_records" not in payload


def test_operator_json_serializes_dataclasses_inside_collections() -> None:
    payload = json.loads(
        as_json([PreflightCheck(name="docker", ok=True, detail="available")])
    )

    assert payload == [{"detail": "available", "name": "docker", "ok": True}]


def test_operator_preview_is_side_effect_free(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    preview = service.preview(ExperimentRequest(experiment_id="demo"))

    assert preview.cells == 1
    assert preview.estimated_trials == 1
    assert preview.harnesses == ("codex",)
    assert len(preview.matrix_cells) == 1
    assert preview.matrix_cells[0].task_id == "task-one"
    assert preview.matrix_cells[0].trial_count == 1
    assert not (tmp_path / ".fugue").exists()


def test_preview_setup_and_execution_resolve_the_same_coordinates(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    request = ExperimentRequest(experiment_id="demo")

    preview = service.resolve_run_plan(request, run_id="preview")
    setup = service.resolve_run_plan(request, run_id="setup")
    materialized = service._materialize_run_plan(
        service.resolve_run_plan(request, run_id="execution"),
        run_id="execution",
    )

    def coordinates(plan):
        return [
            (
                cell.workload_id,
                cell.task_id,
                cell.harness,
                cell.context_system_id,
                cell.trial_index,
                cell.comparison_example_id,
                cell.candidate_id,
                cell.execution_fingerprint,
                cell.applicable,
            )
            for cell in plan.cells
        ]

    assert coordinates(preview) == coordinates(setup) == coordinates(materialized)
    assert preview.cells[0].config_sha256 == ""
    assert materialized.cells[0].config_sha256


def test_run_rejects_incomplete_generated_evaluation_with_planning_command(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    experiment = replace(
        service.experiment("demo"),
        judge_model="openai/gpt-5-mini",
        evaluation_generation=EvaluationGenerationSpec(
            suite_id="missing-suite",
            workload_id="capabilities",
        ),
        workloads=[
            WorkloadSpec(
                id="capabilities",
                runner="harbor",
                manifest=Path("configs/fugue/evaluations/missing-suite/manifest.yaml"),
                scorers=[
                    RubricScorerSelection(
                        type="rubric",
                        path="configs/fugue/evaluations/missing-suite/rubric.yaml",
                    )
                ],
            )
        ],
    )

    with pytest.raises(ValueError, match=r"fugue plan demo"):
        service.prepare_context(
            service.request_for_experiment(experiment),
            experiment=experiment,
        )

    assert not (tmp_path / ".fugue").exists()


def test_context_rebuild_keeps_content_addressed_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    rebuild_values: list[bool] = []
    monkeypatch.setattr(
        operator_module,
        "materialize_manifest_dataset",
        lambda manifest, repo_root, *, rebuild=False: rebuild_values.append(rebuild),
    )
    experiment = replace(
        service.experiment("demo"),
        workloads=[
            WorkloadSpec(
                id="coding",
                runner="harbor",
                manifest=Path("datasets/demo.yaml"),
            )
        ],
    )

    service.prepare_context(
        service.request_for_experiment(experiment),
        rebuild=True,
        experiment=experiment,
    )

    assert rebuild_values == [False]


def test_context_preparation_rejects_harbor_tasks_without_repository_identity(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    (tmp_path / "configs/fugue/context-systems/rag-bm25.yaml").write_text(
        """
id: rag-bm25
title: Fugue RAG
provider: fugue.bench.context:RagContextProvider
version: "2"
capabilities: [prepare, retrieve, bind]
deliveries: [portable]
license: Fugue
config: {mode: bm25}
"""
    )
    (tmp_path / "datasets/demo.yaml").write_text(
        """
dataset: {path: local-tasks}
harnesses: [{name: codex, agent: fugue.agents:FugueCodex}]
tasks: [{id: task-one}]
"""
    )
    experiment = service.experiment("demo")
    experiment = replace(
        experiment,
        variants=[
            replace(
                experiment.variants[0],
                context=ContextSelection(system_id="rag-bm25"),
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="context preparation requires an immutable repository.*task-one",
    ):
        service.prepare_context(
            service.request_for_experiment(experiment),
            experiment=experiment,
        )


def test_setup_materializes_a_dataset_before_preparing_its_task_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        operator_module,
        "materialize_manifest_dataset",
        lambda *args, **kwargs: events.append("dataset"),
    )
    monkeypatch.setattr(operator_module, "agent_runtime_spec", lambda harness: None)
    monkeypatch.setattr(service, "prepare_context", lambda *args, **kwargs: ())

    def prepare_task(manifest, task, **kwargs):
        assert events == ["dataset"]
        events.append("task")
        return {
            "image": "fugue-task:test",
            "image_id": "sha256:" + "a" * 64,
            "recipe_sha256": "b" * 64,
        }

    monkeypatch.setattr(operator_module, "prepare_task_runtime", prepare_task)

    service.prepare(ExperimentRequest(experiment_id="demo"))

    assert events == ["dataset", "task"]


def test_generated_evaluation_preflight_requires_explicit_judge(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    experiment = replace(
        service.experiment("demo"),
        workloads=[
            WorkloadSpec(
                id="capabilities",
                runner="harbor",
                manifest=Path("datasets/demo.yaml"),
                scorers=[
                    RubricScorerSelection(
                        type="rubric",
                        path="configs/fugue/evaluations/generated/rubric.yaml",
                    )
                ],
            )
        ],
    )

    checks = service.preflight(
        service.request_for_experiment(experiment),
        live=False,
        experiment=experiment,
    )

    assert any(
        check.name == "generated evaluation judge model" and not check.ok
        for check in checks
    )


def test_request_for_experiment_keeps_inherited_scale_out_of_overrides(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    experiment = service.experiment("demo")

    request = service.request_for_experiment(experiment)

    assert request.harnesses == ("codex",)
    assert request.variants == ("baseline",)
    assert request.n_attempts is None
    assert request.n_tasks is None
    assert request.n_concurrent is None


def test_request_tags_extend_authored_experiment_tags(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    experiment = replace(service.experiment("demo"), tags=["authored-treatment"])
    request = replace(
        service.request_for_experiment(experiment),
        tags=("campaign:demo", "authored-treatment"),
    )

    selected = operator_module._experiment_with_request_overrides(
        experiment,
        request,
    )

    assert selected.tags == ["authored-treatment", "campaign:demo"]


def test_operator_applies_agent_preset_without_saving(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)

    experiment = service.apply_agent_preset(
        service.experiment("demo"), "demo-maintainer"
    )

    assert experiment.model == "openai/gpt-5"
    assert experiment.harnesses == ["codex"]
    assert [item.id for item in experiment.variants] == ["maintainer-recommended"]
    assert experiment.variants[0].prompt_id == "demo-prompt"
    assert experiment.variants[0].skill_ids == ["demo-skill"]
    assert service.experiment("demo").variants[0].id == "baseline"


def test_execute_run_persists_snapshot_before_first_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    monkeypatch.setattr(operator_module, "agent_runtime_spec", lambda harness: None)
    monkeypatch.setattr(operator_module, "_verify_rendered_setup", lambda jobs: None)
    run_id = "transaction-order"
    observed: list[str] = []

    class FakeLiveEvaluation:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def begin_cell(self, cell):
            return None

        def finish_cell(self, cell, outcome) -> None:
            pass

        def finalize(self) -> PublicationResult:
            return PublicationResult(
                published=1,
                skipped=0,
                evaluations=(
                        PublishedEvaluation(
                            candidate_id="candidate-a",
                            name="Demo evaluation",
                            examples=1,
                            project="team/fugue-experiments",
                            url="https://wandb.ai/team/project/r/call/eval-1",
                        agent_predictions=1,
                        linked_agent_predictions=1,
                    ),
                ),
            )

    monkeypatch.setattr(
        "fugue.bench.operator.LiveEvaluationCoordinator", FakeLiveEvaluation
    )

    def validate(config_paths) -> None:
        assert config_paths
        observed.append("validate")

    monkeypatch.setattr("fugue.bench.operator.validate_harbor_job_configs", validate)

    def runner(command, **kwargs):
        lock_path = tmp_path / ".fugue/runtime" / run_id / "input-lock.json"
        lock = json.loads(lock_path.read_text())
        manifest = read_run_manifest(lock_path.parent)
        assert verify_snapshot(lock)
        assert manifest is not None
        assert manifest["status"] == "running"
        assert manifest["pid"] == os.getpid()
        assert manifest["detached"] is False
        assert service.supervisor.get(run_id).status == "running"
        assert manifest["snapshot_sha256"] == lock["snapshot_sha256"]
        observed.append(command[0])
        return subprocess.CompletedProcess(command, 0)

    result = service.execute_run(
        ExperimentRequest(experiment_id="demo"),
        run_id=run_id,
        cell_runner=runner,
    )

    assert observed == ["validate", "harbor"]
    assert result.status == "passed"
    assert result.evaluation_urls == ("https://wandb.ai/team/project/r/call/eval-1",)
    assert service.run_evaluation(run_id) == result.evaluations[0]
    manifest = read_run_manifest(tmp_path / ".fugue/runtime" / run_id)
    assert manifest is not None
    assert manifest["evaluation_runs"][0]["agent_predictions"] == 1
    assert manifest["evaluation_runs"][0]["linked_agent_predictions"] == 1
    assert manifest["harbor_conformance"]["status"] == "not_applicable"
    assert manifest["harbor_conformance"]["enforced"] is False
    assert (
        tmp_path
        / manifest["harbor_conformance"]["path"]
    ).is_file()


def test_execute_run_records_optional_live_publication_failure_without_changing_task_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_operator_repo(tmp_path)
    monkeypatch.setattr(operator_module, "agent_runtime_spec", lambda harness: None)
    monkeypatch.setattr(operator_module, "_verify_rendered_setup", lambda jobs: None)
    monkeypatch.setattr(
        "fugue.bench.operator.validate_harbor_job_configs", lambda paths: None
    )

    class FailedPublication:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def begin_cell(self, cell):
            return None

        def finish_cell(self, cell, outcome) -> None:
            pass

        def finalize(self) -> PublicationResult:
            return PublicationResult(
                published=0,
                skipped=0,
                failures=("cell-a: five-link evidence did not reconcile",),
            )

    monkeypatch.setattr(
        "fugue.bench.operator.LiveEvaluationCoordinator", FailedPublication
    )

    result = service.execute_run(
        ExperimentRequest(experiment_id="demo"),
        run_id="publication-failure",
        cell_runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0
        ),
    )

    assert result.status == "passed"
    manifest = read_run_manifest(
        tmp_path / ".fugue/runtime/publication-failure"
    )
    assert manifest is not None
    assert manifest["status"] == "passed"
    assert manifest["observability_status"] == "failed"
    assert manifest["evaluation_failures"] == [
        "cell-a: five-link evidence did not reconcile"
    ]
    assert manifest.get("error") is None

    governed = service.execute_run(
        ExperimentRequest(experiment_id="demo"),
        experiment=replace(
            service.experiment("demo"),
            require_live_evidence=True,
        ),
        run_id="governed-publication-failure",
        cell_runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0
        ),
    )

    assert governed.status == "failed"
    governed_manifest = read_run_manifest(
        tmp_path / ".fugue/runtime/governed-publication-failure"
    )
    assert governed_manifest is not None
    assert governed_manifest["status"] == "failed"
    assert "Live evidence publication did not reconcile" in governed_manifest["error"]


def test_execute_run_uses_preset_concurrency_for_the_operator_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    experiment_path = tmp_path / "configs/fugue/experiments/demo.yaml"
    experiment_path.write_text(
        experiment_path.read_text() + "\npresets:\n  parallel:\n    n_concurrent: 4\n"
    )
    monkeypatch.setattr(operator_module, "agent_runtime_spec", lambda harness: None)
    monkeypatch.setattr(operator_module, "_verify_rendered_setup", lambda jobs: None)
    monkeypatch.setattr(
        "fugue.bench.operator.validate_harbor_job_configs", lambda paths: None
    )
    captured: dict[str, int] = {}

    def execute(cells, *, max_workers, **kwargs):
        captured["max_workers"] = max_workers
        return [CellOutcome(cell.id, "passed", returncode=0) for cell in cells]

    monkeypatch.setattr("fugue.bench.operator.execute_cells", execute)

    result = service.execute_run(
        ExperimentRequest(experiment_id="demo", preset="parallel"),
        run_id="preset-concurrency",
    )

    assert result.status == "passed"
    assert captured["max_workers"] == 4
    manifest = read_run_manifest(tmp_path / ".fugue/runtime/preset-concurrency")
    assert manifest is not None
    assert manifest["max_workers"] == 4


def test_execute_run_only_validates_agent_job_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    monkeypatch.setattr(operator_module, "agent_runtime_spec", lambda harness: None)
    monkeypatch.setattr(operator_module, "_verify_rendered_setup", lambda jobs: None)
    request = ExperimentRequest(experiment_id="demo")
    [agent_job] = service.rendered_jobs(request, run_id="validation-fixture")
    dataset = tmp_path / "direct-diagnostic.yaml"
    dataset.write_text("id: direct-diagnostic\n")
    direct_job = replace(
        agent_job,
        config_path=dataset,
        execution_kind="provider_diagnostic",
    )
    observed: list[Path] = []

    class StopAfterValidation(RuntimeError):
        pass

    def validate(paths: list[Path]) -> None:
        observed.extend(paths)
        raise StopAfterValidation

    monkeypatch.setattr(service, "prepare_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        service,
        "rendered_jobs",
        lambda *args, **kwargs: [agent_job, direct_job],
    )
    monkeypatch.setattr(
        "fugue.bench.operator.validate_harbor_job_configs",
        validate,
    )

    with pytest.raises(StopAfterValidation):
        service.execute_run(request, run_id="agent-config-validation")

    assert observed == [agent_job.config_path]


def test_execute_run_cancellation_closes_started_cell_and_cancels_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    monkeypatch.setattr(operator_module, "agent_runtime_spec", lambda harness: None)
    monkeypatch.setattr(operator_module, "_verify_rendered_setup", lambda jobs: None)
    run_id = "operator-cancellation"
    cancellation = threading.Event()
    events: list[tuple[str, object]] = []

    class FakeLiveEvaluation:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def begin_cell(self, cell) -> None:
            events.append(("begin", cell.id))

        def finish_cell(self, cell, outcome) -> None:
            events.append(("finish", outcome.status))

        def cancel_open_predictions(self, reason: str) -> None:
            events.append(("cancel_open", reason))

        def finalize(self, *, cancelled: bool = False) -> PublicationResult:
            events.append(("finalize", cancelled))
            return PublicationResult(published=0, skipped=0)

        def abort(self, reason: str) -> PublicationResult:
            events.append(("abort", reason))
            return PublicationResult(published=0, skipped=0)

    monkeypatch.setattr(
        "fugue.bench.operator.LiveEvaluationCoordinator", FakeLiveEvaluation
    )
    monkeypatch.setattr(
        "fugue.bench.operator.validate_harbor_job_configs", lambda paths: None
    )

    def runner(command, **kwargs):
        cancellation.set()
        return subprocess.CompletedProcess(command, 1)

    result = service.execute_run(
        ExperimentRequest(experiment_id="demo"),
        run_id=run_id,
        cell_runner=runner,
        cancellation_event=cancellation,
    )

    assert result.status == "cancelled"
    assert events[0][0] == "begin"
    assert ("finish", "cancelled") in events
    assert ("finalize", True) in events
    assert not any(name == "cancel_open" for name, _ in events)
    manifest = read_run_manifest(tmp_path / ".fugue/runtime" / run_id)
    assert manifest is not None
    assert manifest["status"] == "cancelled"
    assert manifest["cancelled_cells"] == 1
    assert manifest["failed_cells"] == 0


def test_execute_run_internal_abort_after_terminal_cell_is_not_operator_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    monkeypatch.setattr(operator_module, "agent_runtime_spec", lambda harness: None)
    monkeypatch.setattr(operator_module, "_verify_rendered_setup", lambda jobs: None)
    monkeypatch.setattr(
        "fugue.bench.operator.validate_harbor_job_configs", lambda paths: None
    )
    run_id = "operator-internal-abort"
    cancellation = threading.Event()
    events: list[tuple[str, object]] = []

    class FakeLiveEvaluation:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def begin_cell(self, cell) -> None:
            events.append(("begin", cell.id))

        def finish_cell(self, cell, outcome) -> None:
            events.append(("finish", outcome.status))
            cancellation.set()

        def finalize(self, *, cancelled: bool = False) -> PublicationResult:
            events.append(("finalize", cancelled))
            return PublicationResult(published=0, skipped=0)

        def abort(self, reason: str) -> PublicationResult:
            events.append(("abort", reason))
            return PublicationResult(published=0, skipped=0)

    monkeypatch.setattr(
        "fugue.bench.operator.LiveEvaluationCoordinator", FakeLiveEvaluation
    )

    result = service.execute_run(
        ExperimentRequest(experiment_id="demo"),
        run_id=run_id,
        cell_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
        cancellation_event=cancellation,
        cancellation_origin="internal",
    )

    assert result.status == "failed"
    assert result.cancelled == 0
    assert ("finish", "passed") in events
    assert ("abort", "Run stopped by an internal safety gate.") in events
    assert not any(name == "finalize" for name, _ in events)
    manifest = read_run_manifest(tmp_path / ".fugue/runtime" / run_id)
    assert manifest is not None
    assert manifest["status"] == "failed"
    assert manifest["error"] == "Run stopped by an internal safety gate."
    assert manifest["cancelled_cells"] == 0
    assert manifest["observability_status"] == "failed"
    assert manifest["evaluation_failures"] == []


def test_governed_execution_failure_is_an_internal_abort_not_operator_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    experiment_path = tmp_path / "configs/fugue/experiments/demo.yaml"
    experiment_path.write_text(
        experiment_path.read_text() + "\nrequire_live_evidence: true\n"
    )
    monkeypatch.setattr(operator_module, "agent_runtime_spec", lambda harness: None)
    monkeypatch.setattr(operator_module, "_verify_rendered_setup", lambda jobs: None)
    monkeypatch.setattr(
        "fugue.bench.operator.validate_harbor_job_configs", lambda paths: None
    )
    aborted: list[str] = []

    class FakeLiveEvaluation:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def begin_cell(self, cell) -> None:
            pass

        def finish_cell(self, cell, outcome) -> None:
            pass

        def finalize(self, *, cancelled: bool = False) -> PublicationResult:
            raise AssertionError("an internally aborted run must not finalize")

        def abort(self, reason: str) -> PublicationResult:
            aborted.append(reason)
            return PublicationResult(published=0, skipped=0)

    monkeypatch.setattr(
        "fugue.bench.operator.LiveEvaluationCoordinator", FakeLiveEvaluation
    )

    def fail_governed_cells(cells, **kwargs):
        internal_abort = kwargs.get("internal_abort_event")
        assert isinstance(internal_abort, threading.Event)
        internal_abort.set()
        return [
            CellOutcome(
                cells[0].id,
                "failed",
                error="RuntimeError: cell log reader failed: Broken pipe",
            )
        ]

    monkeypatch.setattr(operator_module, "execute_cells", fail_governed_cells)
    run_id = "governed-execution-abort"

    result = service.execute_run(
        ExperimentRequest(experiment_id="demo"),
        run_id=run_id,
        cell_runner=lambda *args, **kwargs: None,
    )

    assert result.status == "failed"
    assert result.cancelled == 0
    assert len(aborted) == 1
    assert "cell log reader failed: Broken pipe" in aborted[0]
    manifest = read_run_manifest(tmp_path / ".fugue/runtime" / run_id)
    assert manifest is not None
    assert manifest["status"] == "failed"
    assert manifest["internal_abort"] is True
    assert manifest["cancelled_cells"] == 0
    assert manifest["internally_aborted_cells"] == 0
    assert manifest["operator_cancelled_cells"] == 0
    assert "cell log reader failed: Broken pipe" in manifest["internal_abort_reason"]


def test_execute_run_planning_failure_records_starting_failure_without_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    run_id = "planning-failure"

    def fail_render(*args, **kwargs):
        raise ValueError("invalid exact plan")

    monkeypatch.setattr(service, "rendered_jobs", fail_render)

    with pytest.raises(ValueError, match="invalid exact plan"):
        service.execute_run(ExperimentRequest(experiment_id="demo"), run_id=run_id)

    run_dir = tmp_path / ".fugue/runtime" / run_id
    manifest = read_run_manifest(run_dir)
    assert manifest is not None
    assert manifest["status"] == "failed"
    assert manifest["phase"] == "starting"
    assert not (run_dir / "input-lock.json").exists()
    assert not (run_dir / "cells.jsonl").exists()


def test_snapshot_groups_presentation_and_scoring_variants_by_pure_candidate(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    experiment = service.experiment("demo")
    baseline = experiment.variants[0]
    experiment = replace(
        experiment,
        variants=[
            baseline,
            replace(
                baseline,
                id="renamed",
                label="Renamed only",
                verifier={"type": "pytest"},
            ),
        ],
    )
    request = service.request_for_experiment(experiment)
    jobs = service.rendered_jobs(
        request,
        run_id="pure-candidate",
        experiment=experiment,
    )
    cells = plan_cells(jobs, run_id="pure-candidate", run_name="pure candidate")

    snapshot = build_run_snapshot(
        repo_root=tmp_path,
        run_id="pure-candidate",
        experiment=experiment,
        request={"experiment_id": "demo"},
        jobs=jobs,
        cells=cells,
        env=service.env,
    ).to_dict()

    assert len(snapshot["candidates"]) == 1
    assert next(iter(snapshot["candidates"].values()))["harness"] == "codex"
    assert len(snapshot["candidate_runtime"]) == 1
    assert len(snapshot["runtime"]["executions"]) == 2
    assert {item["candidate_id"] for item in snapshot["planned_matrix"]} == set(
        snapshot["candidates"]
    )


def test_snapshot_scopes_task_runtime_locks_to_execution_coordinates(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    experiment = service.experiment("demo")
    request = service.request_for_experiment(experiment)
    [first] = service.rendered_jobs(
        request,
        run_id="multi-task-runtime",
        experiment=experiment,
    )
    execution = {
        **first.resolved_candidate.execution_definition,
        "task_runtime": {
            "task_id": "task-two",
            "image_id": f"sha256:{'b' * 64}",
        },
    }
    second = replace(
        first,
        task_id="task-two",
        job_name=f"{first.job_name}-task-two",
        config_path=first.config_path.with_name("task-two.json"),
        result_path=first.result_path.with_name("task-two.json"),
        comparison_example_id="comparison-task-two",
        resolved_candidate=ResolvedCandidate(
            candidate_id=first.candidate_id,
            execution_fingerprint=stable_digest(execution),
            _definition_json=json.dumps(
                first.resolved_candidate.definition,
                sort_keys=True,
                separators=(",", ":"),
            ),
            _execution_definition_json=json.dumps(
                execution,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )
    second.config_path.write_text(json.dumps(second.config))
    jobs = [first, second]
    cells = plan_cells(jobs, run_id="multi-task-runtime", run_name="multi task")

    snapshot = build_run_snapshot(
        repo_root=tmp_path,
        run_id="multi-task-runtime",
        experiment=experiment,
        request={"experiment_id": "demo"},
        jobs=jobs,
        cells=cells,
        env=service.env,
    ).to_dict()

    assert len(snapshot["candidate_runtime"]) == 1
    assert "task_runtime" not in snapshot["candidate_runtime"][first.candidate_id]
    assert len(snapshot["runtime_locks"]) == 2
    assert {item["execution_fingerprint"] for item in snapshot["runtime_locks"]} == set(
        snapshot["runtime"]["executions"]
    )
    assert {
        (item.get("task_runtime") or {}).get("task_id")
        for item in snapshot["runtime_locks"]
    } == {None, "task-two"}
    assert verify_snapshot(snapshot)


def test_snapshot_v1_records_the_complete_resolved_plan(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    experiment = service.experiment("demo")
    request = replace(
        service.request_for_experiment(experiment), model="wandb/example/model"
    )
    jobs = service.rendered_jobs(
        request,
        run_id="snapshot-v2",
        experiment=experiment,
    )
    cells = plan_cells(jobs, run_id="snapshot-v2", run_name="snapshot v2")
    expected_cells = [
        {
            "attempt_id": cell.attempt_id,
            "attempt_identity": cell.attempt_identity,
            "task_id": cell.task_id,
            "variant_id": cell.variant_id,
            "harness": cell.harness,
            "trial_index": cell.trial_index,
            "candidate_id": cell.candidate_id,
            "execution_fingerprint": cell.execution_fingerprint,
        }
        for cell in cells
    ]
    approved_unsigned = {
        "schema_version": 1,
        "kind": "approved_comparison_execution",
        "expected_cell_count": len(expected_cells),
        "expected_cells_digest": stable_digest(expected_cells),
        "expected_cells": expected_cells,
    }
    approved = {
        **approved_unsigned,
        "lock_digest": stable_digest(approved_unsigned),
    }
    cells = plan_cells(
        jobs,
        run_id="snapshot-v2",
        run_name="snapshot v2",
        approved_comparison=approved,
    )

    route_identity = model_route_identity(jobs[0].route)
    bridge_runtime = {
        "schema_version": 1,
        "image": "ghcr.io/example/bridge@sha256:" + "b" * 64,
        "config_sha256": "c" * 64,
        "target_route": route_identity,
        "resolved_image_id": "sha256:" + "e" * 64,
    }
    snapshot = build_run_snapshot(
        repo_root=tmp_path,
        run_id="snapshot-v2",
        experiment=experiment,
        request={
            "experiment_id": "demo",
            "approved_comparison": approved,
        },
        jobs=jobs,
        cells=cells,
        env=service.env,
        bridge_runtime=bridge_runtime,
    )

    assert isinstance(snapshot, RunSnapshotV1)
    assert snapshot.schema_version == 1
    assert snapshot.source_experiment is not None
    assert snapshot.resolved_experiment_sha256
    assert snapshot.planned_prediction_count == len(cells)
    assert len(snapshot.capability_plan) == len(cells)
    assert {
        item["attempt_id"] for item in snapshot.planned_matrix
    } == {cell.attempt_id for cell in cells}
    assert all(item["attempt_identity"] for item in snapshot.planned_matrix)
    assert snapshot.request["approved_comparison"] == approved
    assert all(
        item["approved_comparison"] == approved
        for item in snapshot.planned_matrix
    )
    [runtime] = snapshot.candidate_runtime.values()
    assert runtime["model_transport"] == {
        "harness": "codex",
        "wire_protocol": "responses",
        "endpoint_kind": "fugue_bridge",
        "upstream_host": "api.inference.wandb.ai",
        "bridge_required": True,
    }
    assert snapshot.runtime["bridge"] == bridge_runtime
    assert verify_snapshot(snapshot.to_dict())
    for unsupported in (2, 3):
        payload = {**snapshot.to_dict(), "schema_version": unsupported}
        assert not verify_snapshot(payload)


def test_evaluation_assets_are_host_only_and_snapshot_records_only_digest(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    experiment = service.experiment("demo")
    request = service.request_for_experiment(experiment)
    jobs = service.rendered_jobs(request, run_id="gold-lock", experiment=experiment)
    jobs = [
        replace(job, expected_evidence_paths=("src/private-gold.py",)) for job in jobs
    ]
    cells = plan_cells(jobs, run_id="gold-lock", run_name="gold lock")
    evaluation_assets = build_evaluation_asset_lock("gold-lock", cells)
    path = write_evaluation_asset_lock(tmp_path, evaluation_assets)

    assert path.stat().st_mode & 0o777 == 0o600
    assert read_evaluation_asset_lock(path) == evaluation_assets
    snapshot = build_run_snapshot(
        repo_root=tmp_path,
        run_id="gold-lock",
        experiment=experiment,
        request={"experiment_id": "demo", "cohort_id": "qualification"},
        jobs=jobs,
        cells=cells,
        env=service.env,
        evaluation_asset_lock_sha256=evaluation_assets.lock_sha256,
    )
    serialized = json.dumps(snapshot.to_dict(), sort_keys=True)

    assert snapshot.evaluation_asset_lock_sha256 == evaluation_assets.lock_sha256
    assert snapshot.cohort_id == "qualification"
    assert "src/private-gold.py" not in serialized
    assert all(
        "expected_evidence_paths" not in job.config.get("fugue", {})
        and "FUGUE_EXPECTED_EVIDENCE_PATHS" not in job.env
        for job in jobs
    )


def test_operator_resolves_source_provenance_once_per_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_operator_repo(tmp_path)
    experiment = replace(service.experiment("demo"), n_attempts=2)
    provenance = {
        "schema_version": 1,
        "kind": "git",
        "commit": "a" * 40,
        "dirty": False,
    }
    calls: list[Path] = []

    def resolve_once(repo_root: Path) -> dict:
        calls.append(repo_root)
        return provenance

    monkeypatch.setattr(
        "fugue.bench.operator.resolve_fugue_source_provenance",
        resolve_once,
    )
    jobs = service.rendered_jobs(
        service.request_for_experiment(experiment),
        run_id="source-once",
        experiment=experiment,
    )

    assert len(jobs) == 2
    assert calls == [tmp_path]
    assert all(
        job.resolved_candidate.execution_definition["fugue_source"] == provenance
        for job in jobs
    )


def test_snapshot_locks_generated_context_runtime_per_cell(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    (tmp_path / "configs/fugue/context-systems/rag-bm25.yaml").write_text(
        """
id: rag-bm25
title: Fugue RAG
provider: fugue.bench.context:RagContextProvider
version: "2"
capabilities: [prepare, retrieve, bind]
deliveries: [portable]
license: Fugue
config:
  mode: bm25
  binding: {managed_runtime: fugue_context}
"""
    )
    experiment = service.experiment("demo")
    experiment = replace(
        experiment,
        variants=[
            replace(
                experiment.variants[0],
                context=ContextSelection(system_id="rag-bm25"),
            )
        ],
    )
    request = service.request_for_experiment(experiment)
    jobs = service.rendered_jobs(
        request,
        run_id="context-lock",
        experiment=experiment,
    )
    cells = plan_cells(jobs, run_id="context-lock", run_name="context lock")

    snapshot = build_run_snapshot(
        repo_root=tmp_path,
        run_id="context-lock",
        experiment=experiment,
        request={"experiment_id": "demo"},
        jobs=jobs,
        cells=cells,
        env=service.env,
    ).to_dict()

    [job] = jobs
    [planned] = snapshot["planned_matrix"]
    asset_id = next(
        item
        for item in planned["generated_runtime_asset_ids"]
        if "/context-runtime/" in snapshot["assets"][item]["path"]
    )
    asset = snapshot["assets"][asset_id]
    runtime_file = next(
        item
        for item in job.generated_runtime_files
        if "context-runtime" in item.as_posix()
    )
    raw = runtime_file.read_bytes()
    assert asset["kind"] == "generated_runtime"
    assert asset["path"].startswith(".fugue/runtime/context-lock/context-runtime/")
    assert asset["sha256"] == hashlib.sha256(raw).hexdigest()
    assert asset["body"].encode() == raw
    assert (
        snapshot["candidate_runtime"][job.candidate_id]["context_runtime"]
        == (job.resolved_candidate.execution_definition["context_runtime"])
    )
    fugue_source = job.resolved_candidate.execution_definition["fugue_source"]
    assert fugue_source["kind"] == "unversioned"
    assert fugue_source["dirty"] is True
    assert snapshot["runtime"]["fugue_source"] == fugue_source
    assert snapshot["candidate_runtime"][job.candidate_id]["fugue_source"] == (
        fugue_source
    )
    assert verify_snapshot(snapshot)
    tampered = json.loads(json.dumps(snapshot))
    tampered["assets"][asset_id]["body"] += "\n# changed after planning\n"
    assert not verify_snapshot(tampered)
    assert "model-secret" not in json.dumps(snapshot)


def test_snapshot_locks_generated_integration_runtime_per_cell(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir()
    (root / "api.yaml").write_text(
        """
id: api
version: "1"
support: experimental
runtime:
  type: compose
  image: ghcr.io/example/api@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  service: api
  port: 8000
interfaces:
  - {type: http, name: endpoint, path: /}
"""
    )
    experiment = service.experiment("demo")
    experiment = replace(
        experiment,
        variants=[
            replace(
                experiment.variants[0],
                integrations=[IntegrationSelection("api")],
            )
        ],
    )
    request = service.request_for_experiment(experiment)
    jobs = service.rendered_jobs(
        request,
        run_id="integration-lock",
        experiment=experiment,
    )
    cells = plan_cells(jobs, run_id="integration-lock", run_name="integration lock")

    snapshot = build_run_snapshot(
        repo_root=tmp_path,
        run_id="integration-lock",
        experiment=experiment,
        request={"experiment_id": "demo"},
        jobs=jobs,
        cells=cells,
        env=service.env,
    ).to_dict()

    [job] = jobs
    runtime_file = next(
        item
        for item in job.generated_runtime_files
        if "integrations" in item.as_posix()
    )
    asset_id = next(
        item
        for item in snapshot["planned_matrix"][0]["generated_runtime_asset_ids"]
        if "/integrations/" in snapshot["assets"][item]["path"]
    )
    asset = snapshot["assets"][asset_id]
    assert asset["kind"] == "generated_runtime"
    assert asset["path"].startswith(".fugue/runtime/integration-lock/integrations/")
    assert asset["body"] == runtime_file.read_text()
    assert asset["sha256"] == hashlib.sha256(runtime_file.read_bytes()).hexdigest()
    assert "api:" in asset["body"]
    assert job.resolved_candidate.definition["integrations"][0]["id"] == "api"
    assert job.integration_provenance[0]["support"] == "experimental"
    assert verify_snapshot(snapshot)


def test_candidate_prefixes_are_display_only_and_must_resolve_uniquely(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    run_id = "candidate-prefixes"
    first = "abc" + "1" * 61
    second = "abc" + "2" * 61
    run_dir = tmp_path / ".fugue/runtime" / run_id
    run_dir.mkdir(parents=True)
    write_run_manifest(
        tmp_path,
        run_id,
        {"status": "passed", "run_name": "prefixes", "experiment_id": "demo"},
    )
    (run_dir / "input-lock.json").write_text(
        json.dumps(
            {
                "candidates": {
                    first: {"harness": "codex"},
                    second: {"harness": "codex"},
                },
                "planned_matrix": [],
            }
        )
    )

    summary = service.run_summary(run_id)

    assert {item.candidate_id for item in summary.candidates} == {first, second}
    assert all(len(item.display_id) == 12 for item in summary.candidates)
    assert service.resolve_candidate_id(run_id, first[:12]) == first
    with pytest.raises(ValueError, match="ambiguous"):
        service.resolve_candidate_id(run_id, "abc")


def test_candidate_summary_separates_execution_and_benchmark_outcomes(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    run_id = "candidate-outcomes"
    candidate_id = "a" * 64
    run_dir = tmp_path / ".fugue/runtime" / run_id
    run_dir.mkdir(parents=True)
    write_run_manifest(
        tmp_path,
        run_id,
        {"status": "passed", "run_name": "outcomes", "experiment_id": "demo"},
    )
    (run_dir / "input-lock.json").write_text(
        json.dumps(
            {
                "candidates": {candidate_id: {"harness": "codex"}},
                "candidate_runtime": {candidate_id: {"harness": "codex"}},
                "planned_matrix": [
                    {"cell_id": "pass", "candidate_id": candidate_id},
                    {"cell_id": "fail", "candidate_id": candidate_id},
                ],
            }
        )
    )
    (run_dir / "cells.jsonl").write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {
                    "cell_id": "pass",
                    "candidate_id": candidate_id,
                    "status": "passed",
                    "benchmark_outcome": "passed",
                    "reward": 1.0,
                },
                {
                    "cell_id": "fail",
                    "candidate_id": candidate_id,
                    "status": "passed",
                    "benchmark_outcome": "failed",
                    "reward": 0.0,
                },
            )
        )
        + "\n"
    )

    summary = service.run_summary(run_id)
    [candidate] = summary.candidates

    assert summary.status == "passed"
    assert candidate.passed == 1
    assert candidate.failed == 1
    assert candidate.execution_failed == 0
    assert candidate.unscored == 0
    assert candidate.completeness == 1.0
    assert candidate.packageable is False
    assert "1 failed benchmark cell(s)" in candidate.packageability_reason


def test_run_summary_keeps_cancellation_separate_from_failures(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    run_id = "cancelled-summary"
    candidate_id = "c" * 64
    run_dir = tmp_path / ".fugue/runtime" / run_id
    run_dir.mkdir(parents=True)
    write_run_manifest(
        tmp_path,
        run_id,
        {
            "status": "cancelled",
            "run_name": "cancelled",
            "experiment_id": "demo",
            "cancellation_cleanup_status": "passed",
            "cancellation_cleanup_projects": ["fugue-run-cell"],
            "cancellation_cleanup_errors": [],
        },
    )
    (run_dir / "input-lock.json").write_text(
        json.dumps(
            {
                "candidates": {candidate_id: {"harness": "codex"}},
                "planned_matrix": [
                    {"cell_id": "cancelled", "candidate_id": candidate_id},
                    {"cell_id": "interrupted", "candidate_id": candidate_id},
                    {"cell_id": "failed", "candidate_id": candidate_id},
                ],
            }
        )
    )
    (run_dir / "cells.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "cell_id": cell_id,
                    "candidate_id": candidate_id,
                    "status": status,
                    "benchmark_outcome": "unscored",
                }
            )
            for cell_id, status in (
                ("cancelled", "cancelled"),
                ("interrupted", "interrupted"),
                ("failed", "failed"),
            )
        )
        + "\n"
    )

    summary = service.run_summary(run_id)
    [candidate] = summary.candidates

    assert summary.failed == 1
    assert summary.cancelled == 1
    assert summary.interrupted == 1
    assert summary.cancellation_cleanup_status == "passed"
    assert summary.cancellation_cleanup_projects == ("fugue-run-cell",)
    assert summary.cancellation_cleanup_errors == ()
    assert candidate.execution_failed == 1
    assert candidate.cancelled == 1
    assert candidate.interrupted == 1


def test_run_summary_preserves_not_applicable_reason(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    run_id = "not-applicable-reason"
    candidate_id = "b" * 64
    run_dir = tmp_path / ".fugue/runtime" / run_id
    run_dir.mkdir(parents=True)
    write_run_manifest(
        tmp_path,
        run_id,
        {"status": "passed", "run_name": "n-a", "experiment_id": "demo"},
    )
    (run_dir / "input-lock.json").write_text(
        json.dumps(
            {
                "candidates": {candidate_id: {"harness": "sequence"}},
                "planned_matrix": [{"cell_id": "latmd", "candidate_id": candidate_id}],
            }
        )
    )
    (run_dir / "cells.jsonl").write_text(
        json.dumps(
            {
                "cell_id": "latmd",
                "candidate_id": candidate_id,
                "status": "not_applicable",
                "benchmark_outcome": "not_applicable",
                "skip_reason": "LAT_LLM_KEY is missing",
            }
        )
        + "\n"
    )

    [cell] = service.run_summary(run_id).cells

    assert cell.skip_reason == "LAT_LLM_KEY is missing"


def test_multi_file_plan_save_cleans_new_assets_when_commit_marker_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_operator_repo(tmp_path)
    experiment = service.experiment("demo")
    assets = (
        AssetDraft("prompt", "new-prompt", "New prompt", "# Prompt\n"),
        AssetDraft("skill", "new-skill", "New skill", "# Skill\n"),
    )

    def fail_save(*args, **kwargs):
        raise OSError("experiment commit marker failed")

    monkeypatch.setattr("fugue.bench.operator.save_experiment_data", fail_save)

    with pytest.raises(OSError, match="commit marker"):
        service.save_working_experiment(
            experiment,
            service.request_for_experiment(experiment),
            experiment_id="save-failure",
            assets=assets,
        )

    assert not (tmp_path / "configs/fugue/prompts/new-prompt.md").exists()
    assert not (tmp_path / "configs/fugue/skills/new-skill").exists()
    assert not (tmp_path / "configs/fugue/experiments/save-failure.yaml").exists()


def test_plan_save_validates_evaluation_assets_before_writing(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    experiment = service.experiment("demo")
    invalid = AssetDraft(
        "evaluation_rubric",
        "invalid-suite",
        "rubric.yaml",
        "schema_version: 1\nid: invalid-suite\ndimensions: []\n",
    )

    with pytest.raises(ValueError, match="dimensions are required"):
        service.save_working_experiment(
            experiment,
            service.request_for_experiment(experiment),
            experiment_id="invalid-assets",
            assets=(invalid,),
        )

    assert not (tmp_path / "configs/fugue/evaluations/invalid-suite").exists()


def test_start_bridge_loads_the_requested_experiment(
    tmp_path: Path, monkeypatch
) -> None:
    service = make_operator_repo(tmp_path)
    captured: dict[str, object] = {}

    def fake_bridge_up(target, **kwargs):
        captured.update({"target": target, **kwargs})
        return object()

    monkeypatch.setattr("fugue.bench.operator.bridge_up", fake_bridge_up)

    service.start_bridge(ExperimentRequest(experiment_id="demo"))

    assert captured["target"] == "openai/gpt-5"
    assert captured["builder_model"] is None
    assert captured["judge_model"] is None


def test_prepare_skills_only_inspects_selected_remote_sources(
    tmp_path: Path, monkeypatch
) -> None:
    service = make_operator_repo(tmp_path)
    source_root = tmp_path / "configs" / "fugue" / "skill-sources"
    source_root.mkdir(parents=True)
    (source_root / "remote.yaml").write_text(
        """
id: remote
source:
  type: git
  url: https://github.com/example/skills
  ref: 0000000000000000000000000000000000000000
  path: skills/remote
"""
    )
    (tmp_path / "configs" / "fugue" / "experiments" / "demo.yaml").write_text(
        """
id: demo
title: Demo
manifest: datasets/demo.yaml
model: openai/gpt-5
harnesses: [codex]
variants:
  - {id: remote, label: Remote, skills: [remote], context: {system_id: none, delivery: portable}}
"""
    )
    calls: list[tuple[str, bool]] = []

    def fake_prepare(skill_id, repo_root, *, refresh=False):
        assert repo_root == tmp_path
        calls.append((skill_id, refresh))
        return skill_id

    monkeypatch.setattr("fugue.bench.operator.prepare_skill_source", fake_prepare)

    records = service.prepare_skills(
        ExperimentRequest(experiment_id="demo"), refresh=True
    )

    assert records == ("remote",)
    assert calls == [("remote", True)]


def test_ephemeral_experiment_launch_persists_runtime_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    service = make_operator_repo(tmp_path)

    def start_detached(**values):
        write_run_manifest(
            tmp_path,
            values["run_id"],
            {
                "status": "starting",
                "run_name": values["run_name"],
                "experiment_id": values["experiment_id"],
                "combined_log": str(
                    tmp_path / ".fugue/runtime" / values["run_id"] / "combined.log"
                ),
            },
        )
        return service.supervisor.get(values["run_id"], recover=False)

    monkeypatch.setattr(service.supervisor, "start_detached", start_detached)
    experiment = service.experiment("demo")
    run = service.launch(
        ExperimentRequest(experiment_id="demo"),
        experiment=experiment,
        run_id="campaign-run-1",
    )

    snapshot = tmp_path / ".fugue/runtime" / run.run_id / "experiment.yaml"
    assert run.run_id == "campaign-run-1"
    assert snapshot.is_file()
    assert "id: demo" in snapshot.read_text()


def test_run_links_use_the_project_recorded_at_launch(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    write_run_manifest(
        tmp_path,
        "run-original-project",
        {
            "status": "passed",
            "run_name": "Original project",
            "experiment_id": "demo",
            "trace_project": "other-team/original-project",
            "evaluation_runs": [
                {
                    "candidate_id": "candidate-a",
                    "name": "Original evaluation",
                    "examples": 2,
                    "url": "https://wandb.ai/other-team/original-project/r/call/eval-1",
                    "linked_predictions": 2,
                }
            ],
        },
    )

    links = service.run_links("run-original-project")

    assert links.agents == ("https://wandb.ai/other-team/original-project/weave/agents")
    assert service.run_evaluation("run-original-project").url == (
        "https://wandb.ai/other-team/original-project/r/call/eval-1"
    )


def test_operator_results_prefers_enriched_normalized_exports(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "demo.jsonl").write_text(
        json.dumps(
            {
                "record_type": "trial",
                "run_id": "run-1",
                "candidate_id": "candidate-1",
                "comparison_example_id": "example-1",
                "trial_index": 1,
                "run_key": "run-1:task:codex:trial-1",
                "harness": "codex",
                "experiment_id": "demo",
                "variant_id": "baseline",
                "context_system_id": "none",
                "model": "openai/gpt-5",
                "pass": True,
                "reward": 0.8,
                "wall_time_sec": 4.0,
                "cost_usd": 0.02,
                "n_input_tokens": 100,
                "n_output_tokens": 20,
                "weave_agent_name": "codex",
                "weave_conversation_ids": ["conversation-1"],
                "weave_turn_count": 1,
                "weave_tool_call_count": 3,
            }
        )
        + "\n"
    )

    result = service.results()

    assert result.total == 1
    assert result.pass_rate == 1.0
    assert result.average_reward == 0.8
    assert result.average_wall_time_sec == 4.0
    assert result.tool_calls == 3
    assert result.turns == 1
    assert result.agent_traces[0].conversation_ids == ("conversation-1",)
