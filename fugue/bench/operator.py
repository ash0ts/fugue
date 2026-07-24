from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fugue.bench.agent_runtime import (
    prepare_runtime as prepare_agent_runtime,
)
from fugue.bench.agent_runtime import runtime_ready as agent_runtime_ready
from fugue.bench.agent_runtime import runtime_spec as agent_runtime_spec
from fugue.bench.candidates import (
    CANDIDATE_IDENTITY_SCHEMA_VERSION,
    comparison_example_id,
    resolve_candidate,
)
from fugue.bench.context import (
    CONTEXT_MANIFEST,
    DEFAULT_CACHE_ROOT,
    ContextRuntime,
    ContextSystemSpec,
    RepositorySnapshot,
    checkout_repository,
    get_context_system,
    list_context_systems,
    preflight_context,
)
from fugue.bench.context import (
    prepare_context as build_context,
)
from fugue.bench.context_contracts import resolve_context_capabilities
from fugue.bench.datasets import materialize_manifest_dataset
from fugue.bench.evaluation_assets import (
    attach_evaluation_assets,
    load_evaluation_gold_patch,
    prepare_evaluation_assets,
)
from fugue.bench.evaluations import evaluation_asset_path, load_cases, load_rubric
from fugue.bench.execution import (
    PlannedCell,
    execute_cells,
    latest_cell_records,
    mark_unfinished_cells,
    new_run_id,
    plan_cells,
    update_run_manifest,
    write_run_manifest,
)
from fugue.bench.export import (
    GeneratedEvaluationCoordinator,
    LiveEvaluationCoordinator,
    PublishedEvaluation,
    compile_export,
    export_rows,
    normalize_prediction_rows,
    write_jsonl,
)
from fugue.bench.job_config import RenderedJob, preview_jobs, render_jobs
from fugue.bench.library import (
    ExperimentSpec,
    FeatureVariant,
    PresetSpec,
    WorkloadSpec,
    experiment_from_data,
    experiment_to_yaml,
    experiment_with_overrides,
    get_agent_preset,
    get_experiment,
    list_agent_presets,
    list_experiments,
    save_experiment_data,
    scorer_reference,
    validate_id,
)
from fugue.bench.manifest import (
    BenchmarkManifest,
    FixtureRepositorySpec,
    fixture_repository_digest,
    load_manifest,
)
from fugue.bench.portable_runtime import prepare_runtime as prepare_portable_runtime
from fugue.bench.portable_runtime import runtime_ready as portable_runtime_ready
from fugue.bench.reproducibility import (
    build_evaluation_asset_lock,
    build_run_snapshot,
    write_evaluation_asset_lock,
    write_run_input_lock,
)
from fugue.bench.runtime_manager import prepare_runtime, runtime_ready, runtime_spec
from fugue.bench.runtime_provenance import resolve_fugue_source_provenance
from fugue.bench.scoring import read_treatment_selection_lock
from fugue.bench.services import (
    GRAPHITI_SERVICE_ID,
    ManagedServiceStatus,
    managed_service_environment,
    managed_service_statuses,
    managed_services_for_systems,
    start_managed_services,
    stop_managed_services,
    without_managed_service_environment,
)
from fugue.bench.sources import (
    SkillInspection,
    SkillLockEntry,
    approve_skill_source,
    list_skill_source_ids,
    prepare_skill_source,
)
from fugue.bench.supervisor import ManagedRun, RunSupervisor
from fugue.bench.task_runtime import (
    prepare_task_runtime,
    read_task_runtime_lock,
    task_architecture,
    task_runtime_requires_verification,
)
from fugue.bench.workloads import (
    PreparedWorkloadDataset,
    load_workload_dataset,
    prepare_workload_dataset,
)
from fugue.bridge import (
    BridgeFiles,
    bridge_runtime_attestation,
    bridge_status,
    bridge_up,
)
from fugue.model_plane import (
    model_route_identity,
    resolve_harness_model_route,
    resolve_model_route,
    select_model,
    trace_env_defaults,
    trace_project_slug,
)
from fugue.weave_support import trace_async_operation

if TYPE_CHECKING:
    from fugue.bench.ai import AnalysisPreview, AnalysisResult, AnalysisSpec
from fugue.preflight import PreflightCheck, validate_harbor_job_configs


@dataclass(frozen=True)
class ExperimentRequest:
    experiment_id: str = "pilot"
    manifest: Path | None = None
    preset: str | None = None
    workloads: tuple[str, ...] = ()
    harnesses: tuple[str, ...] = ()
    systems: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    model: str | None = None
    builder_model: str | None = None
    judge_model: str | None = None
    n_attempts: int | None = None
    n_tasks: int | None = None
    n_concurrent: int | None = None
    run_name: str | None = None
    tags: tuple[str, ...] = ()
    jobs_dir: Path | None = None
    trace_content: str | None = None
    agent_preset_id: str | None = None
    cohort_id: str | None = None
    selection_lock: Path | None = None


@dataclass(frozen=True)
class PreviewCellSummary:
    harness: str
    variant_id: str
    variant_label: str
    context_system_id: str
    workload_id: str
    task_id: str
    trial_count: int
    applicable: bool
    reason: str | None = None
    context_cache_ready: bool = False
    context_delivery: str = "portable"


@dataclass(frozen=True)
class PreviewSummary:
    cells: int
    applicable_cells: int
    estimated_trials: int
    harnesses: tuple[str, ...]
    variants: tuple[str, ...]
    systems: tuple[str, ...]
    workloads: tuple[str, ...]
    commands: tuple[str, ...]
    matrix_cells: tuple[PreviewCellSummary, ...] = ()


@dataclass(frozen=True)
class CellSummary:
    cell_id: str
    candidate_id: str
    status: str
    harness: str
    variant_id: str
    context_system_id: str
    workload_id: str
    task_id: str
    wall_time_sec: float | None = None
    error: str | None = None
    skip_reason: str | None = None
    context_delivery: str = "portable"
    benchmark_outcome: str = "unscored"
    reward: float | None = None


@dataclass(frozen=True)
class AgentTraceRef:
    agent_name: str
    conversation_ids: tuple[str, ...] = ()
    trace_ids: tuple[str, ...] = ()
    root_span_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateSummary:
    candidate_id: str
    display_id: str
    configuration: dict[str, Any]
    passed: int
    failed: int
    execution_failed: int
    cancelled: int
    interrupted: int
    unscored: int
    pending: int
    not_applicable: int
    completeness: float
    packageable: bool
    packageability_reason: str


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    run_name: str
    experiment_id: str
    status: str
    created_at: str | None
    cells: tuple[CellSummary, ...]
    passed: int
    failed: int
    cancelled: int
    interrupted: int
    pending: int
    not_applicable: int
    candidates: tuple[CandidateSummary, ...]
    log_path: Path
    observability_status: str | None = None
    evaluations: tuple[PublishedEvaluation, ...] = ()
    evaluation_failures: tuple[str, ...] = ()
    cancellation_cleanup_status: str | None = None
    cancellation_cleanup_projects: tuple[str, ...] = ()
    cancellation_cleanup_errors: tuple[str, ...] = ()

    @property
    def evaluation_urls(self) -> tuple[str, ...]:
        return tuple(item.url for item in self.evaluations if item.url)


@dataclass(frozen=True)
class ResultSummary:
    total: int
    passed: int
    failed: int
    pass_rate: float | None
    cost_usd: float
    input_tokens: int
    output_tokens: int
    average_reward: float | None
    average_wall_time_sec: float | None
    tool_calls: int
    turns: int
    context_assigned: int = 0
    context_invoked: int = 0
    context_registered: int = 0
    runtime_mismatched: int = 0
    attributed_errors: int = 0
    linked_traces: int = 0
    unlinked_traces: int = 0
    usage_unavailable: int = 0
    rows: tuple[dict[str, Any], ...] = ()
    agent_traces: tuple[AgentTraceRef, ...] = ()


@dataclass(frozen=True)
class DeepLinks:
    project: str
    weave: str
    agents: str
    trace: str | None = None


@dataclass(frozen=True)
class ExportSummary:
    path: Path
    rows: int
    measurement_path: Path | None = None
    measurements: int = 0
    published: int = 0
    skipped: int = 0
    evaluations: tuple[PublishedEvaluation, ...] = ()
    publication_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextPreparation:
    system_id: str
    task_id: str
    status: str
    detail: str
    cache_key: str | None = None
    path: Path | None = None
    variant_id: str | None = None
    config_digest: str | None = None
    retrieval_mode: str | None = None


@dataclass(frozen=True)
class ContextPreparationTarget:
    variant_id: str
    spec: ContextSystemSpec
    delivery: str
    config_digest: str
    snapshot: RepositorySnapshot


@dataclass(frozen=True)
class AgentRuntimePreparation:
    harness: str
    architecture: str
    status: str
    image: str
    image_id: str
    recipe_sha256: str


@dataclass(frozen=True)
class TaskRuntimePreparation:
    task_id: str
    architecture: str
    status: str
    image: str
    image_id: str
    recipe_sha256: str
    verification_required: bool = False
    verification: dict[str, Any] | None = None


@dataclass(frozen=True)
class SetupPreparation:
    context: tuple[ContextPreparation, ...]
    agent_runtimes: tuple[AgentRuntimePreparation, ...]
    task_runtimes: tuple[TaskRuntimePreparation, ...] = ()
    workload_datasets: tuple[PreparedWorkloadDataset, ...] = ()
    evaluation_asset_locks: tuple[str, ...] = ()
    portable_context_runtime: AgentRuntimePreparation | None = None


@dataclass(frozen=True)
class ResolvedRunPlan:
    request: ExperimentRequest
    experiment: ExperimentSpec
    preset: PresetSpec
    workloads: tuple[WorkloadSpec, ...]
    jobs: tuple[RenderedJob, ...]
    cells: tuple[PlannedCell, ...]
    run_name: str
    max_workers: int


@dataclass(frozen=True)
class ModelRoleStatus:
    role: str
    model: str
    provider: str
    key_env: str
    key_present: bool


@dataclass(frozen=True)
class OperatorStatus:
    trace_project: str
    links: DeepLinks
    model: str
    model_provider: str
    model_key_env: str
    model_key_present: bool
    trace_key_present: bool
    docker_present: bool
    harbor_present: bool
    bridge_ready: bool
    trace_content: str
    experiments: int
    routes: tuple[ModelRoleStatus, ...] = ()
    selected_context_systems: tuple[str, ...] = ()
    context_system_count: int = 0
    context_cache_entries: int = 0
    keys: dict[str, bool] = field(default_factory=dict)


class OperatorService:
    def __init__(
        self,
        repo_root: Path | None = None,
        env_file: Path | None = None,
    ) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.env_file = env_file or self.repo_root / ".env"
        self.supervisor = RunSupervisor(self.repo_root)

    @property
    def env(self) -> dict[str, str]:
        return load_env(self.env_file)

    def status(
        self,
        request: ExperimentRequest | None = None,
        *,
        experiment: ExperimentSpec | None = None,
    ) -> OperatorStatus:
        env = self.env
        selected = experiment or self.experiment(
            request.experiment_id if request else "pilot"
        )
        if request:
            selected = _experiment_with_request_overrides(selected, request)
        selected_model = select_model(
            request.model if request else None,
            env=env,
            experiment_model=selected.model,
        )
        route = resolve_model_route(selected_model, env)
        builder_model = (
            (request.builder_model if request else None)
            or selected.builder_model
            or env.get("FUGUE_BUILDER_MODEL")
            or selected_model
        )
        judge_model = (
            (request.judge_model if request else None)
            or selected.judge_model
            or env.get("FUGUE_JUDGE_MODEL")
        )
        routes = [self._role_status("target", selected_model, env)]
        routes.append(self._role_status("builder", builder_model, env))
        if judge_model:
            routes.append(self._role_status("judge", judge_model, env))
        from fugue.assistant import select_assistant_model

        routes.append(
            self._role_status(
                "composer",
                select_assistant_model(
                    "composer", experiment_model=selected_model, env=env
                ),
                env,
            )
        )
        routes.append(
            self._role_status(
                "analyst",
                select_assistant_model(
                    "composer", experiment_model=selected_model, env=env
                ),
                env,
            )
        )
        trace_project = trace_project_slug(env)
        bridge = bridge_status()
        trace_content = (
            request.trace_content if request else None
        ) or selected.trace_content
        key_names = ("WANDB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        preset = select_preset(selected, request.preset if request else None)
        workloads = select_workloads(
            selected,
            preset,
            list(request.workloads) if request and request.workloads else None,
        )
        selected_systems = tuple(
            dict.fromkeys(
                system_id
                for workload in workloads
                for system_id in (
                    selected_system_ids(
                        selected,
                        workload,
                        preset,
                        (
                            list(request.systems)
                            if request and request.systems
                            else (
                                [
                                    variant.context.system_id
                                    for variant in selected.variants
                                ]
                                if request and request.variants
                                else None
                            )
                        ),
                    )
                    or []
                )
            )
        ) or tuple(
            dict.fromkeys(
                variant.context.system_id
                for variant in selected.variants
                if variant.enabled
            )
        )
        cache_root = self.repo_root / DEFAULT_CACHE_ROOT
        return OperatorStatus(
            trace_project=trace_project,
            links=self.deep_links(),
            model=route.display_model,
            model_provider=route.provider,
            model_key_env=route.api_key_env,
            model_key_present=bool(env.get(route.api_key_env, "").strip()),
            trace_key_present=bool(env.get("WANDB_API_KEY", "").strip()),
            docker_present=shutil.which("docker") is not None,
            harbor_present=shutil.which("harbor") is not None,
            bridge_ready=bool(bridge.get("ok")),
            trace_content=trace_content,
            experiments=len(list_experiments(self.repo_root)),
            routes=tuple(routes),
            selected_context_systems=selected_systems,
            context_system_count=len(list_context_systems(self.repo_root)),
            context_cache_entries=(
                sum(1 for _ in cache_root.glob(f"*/{CONTEXT_MANIFEST}"))
                if cache_root.is_dir()
                else 0
            ),
            keys={key: bool(env.get(key, "").strip()) for key in key_names},
        )

    def preflight(
        self,
        request: ExperimentRequest,
        *,
        live: bool = True,
        experiment: ExperimentSpec | None = None,
    ) -> tuple[PreflightCheck, ...]:
        from fugue.preflight import PreflightCheck, run_preflight

        env = self.env
        external_graphiti_uri = bool(env.get("FUGUE_GRAPHITI_URI", "").strip())
        selected = experiment or self.experiment(request.experiment_id)
        selected = _experiment_with_request_overrides(selected, request)
        target_model = select_model(
            request.model,
            env=env,
            experiment_model=selected.model,
        )
        builder_model = (
            request.builder_model
            or selected.builder_model
            or env.get("FUGUE_BUILDER_MODEL")
            or target_model
        )
        judge_model = (
            request.judge_model or selected.judge_model or env.get("FUGUE_JUDGE_MODEL")
        )
        checks = run_preflight(
            target_model,
            repo_root=self.repo_root,
            env=env,
            live=live,
            harnesses=tuple(selected.harnesses),
            builder_model=builder_model,
            judge_model=judge_model,
        )
        for role, model in (("builder", builder_model), ("judge", judge_model)):
            if not model:
                continue
            try:
                route = resolve_model_route(model, env)
                present = bool(env.get(route.api_key_env, "").strip())
                detail = (
                    f"{route.display_model} can use {route.api_key_env}"
                    if present
                    else f"{route.display_model} requires {route.api_key_env}"
                )
            except ValueError as exc:
                present = False
                detail = str(exc)
            checks.append(PreflightCheck(f"{role} model", present, detail))
        trace_content = request.trace_content or selected.trace_content
        if trace_content == "metadata" and "claude-code" in selected.harnesses:
            checks.append(
                PreflightCheck(
                    "Claude Code trace content",
                    False,
                    "weave-claude-code cannot guarantee metadata-only capture",
                )
            )
        preset = select_preset(selected, request.preset)
        selected_workloads = select_workloads(
            selected,
            preset,
            list(request.workloads) or None,
        )
        generated_scoring = any(
            any(
                not scorer_reference(scorer).startswith("builtin:")
                for scorer in workload.scorers
            )
            for workload in selected_workloads
        )
        if generated_scoring and not (request.judge_model or selected.judge_model):
            checks.append(
                PreflightCheck(
                    "generated evaluation judge model",
                    False,
                    "set judge_model in the experiment or pass --judge-model; "
                    "environment fallback is not accepted for generated rubrics",
                )
            )
        systems = list(request.systems) or (
            _request_variant_system_ids(selected, request)
            if request.variants
            else preset.systems
            or [
                variant.context.system_id
                for variant in selected.variants
                if variant.enabled
            ]
        )
        if "graphiti" in systems:
            env = managed_service_environment(env, repo_root=self.repo_root)
        runtime = ContextRuntime(
            repo_root=self.repo_root,
            cache_root=self.repo_root / DEFAULT_CACHE_ROOT,
            env=env,
        )
        for system_id in dict.fromkeys(systems):
            spec = get_context_system(system_id, self.repo_root)
            for item in asyncio.run(preflight_context(spec, runtime)):
                checks.append(
                    PreflightCheck(
                        f"context {system_id}: {item.name}",
                        item.ok,
                        item.detail,
                    )
                )
        service_specs = tuple(
            service
            for service in managed_services_for_systems(systems)
            if not (external_graphiti_uri and service.id == GRAPHITI_SERVICE_ID)
        )
        for service in managed_service_statuses(
            service_specs,
            repo_root=self.repo_root,
        ):
            checks.append(
                PreflightCheck(
                    f"managed service {service.service_id}",
                    service.ready,
                    f"{service.state}: {service.detail}",
                )
            )
        if preset.id == "full":
            for workload in select_workloads(selected, preset, None):
                if not workload.dataset:
                    continue
                dataset = load_workload_dataset(
                    _resolve(self.repo_root, Path(workload.dataset))
                )
                command = (dataset.source.get("materialize_command") or [None])[0]
                if command:
                    available = shutil.which(str(command)) is not None
                    checks.append(
                        PreflightCheck(
                            f"workload {workload.id} materializer",
                            available,
                            f"{command} is available"
                            if available
                            else f"install {command} before the full preset",
                        )
                    )
            if not judge_model:
                checks.append(
                    PreflightCheck(
                        "judge model",
                        False,
                        "full preset requires a judge model",
                    )
                )
        return tuple(checks)

    def start_bridge(
        self,
        request: ExperimentRequest,
        *,
        experiment: ExperimentSpec | None = None,
    ) -> BridgeFiles:
        env = self.env
        selected = experiment or self.experiment(request.experiment_id)
        target = select_model(
            request.model,
            env=env,
            experiment_model=selected.model,
        )
        return bridge_up(
            target,
            repo_root=self.repo_root,
            env=env,
            builder_model=(
                request.builder_model
                or selected.builder_model
                or env.get("FUGUE_BUILDER_MODEL")
            ),
            judge_model=(
                request.judge_model
                or selected.judge_model
                or env.get("FUGUE_JUDGE_MODEL")
            ),
        )

    def service_status(
        self,
        request: ExperimentRequest,
        *,
        experiment: ExperimentSpec | None = None,
    ) -> tuple[ManagedServiceStatus, ...]:
        selected = experiment or self.experiment(request.experiment_id)
        selected = _experiment_with_request_overrides(selected, request)
        specs = managed_services_for_systems(
            _selected_request_system_ids(selected, request)
        )
        return managed_service_statuses(specs, repo_root=self.repo_root)

    def start_services(
        self,
        request: ExperimentRequest,
        *,
        experiment: ExperimentSpec | None = None,
    ) -> tuple[ManagedServiceStatus, ...]:
        selected = experiment or self.experiment(request.experiment_id)
        selected = _experiment_with_request_overrides(selected, request)
        specs = managed_services_for_systems(
            _selected_request_system_ids(selected, request)
        )
        return start_managed_services(
            specs,
            repo_root=self.repo_root,
            env=self.env,
        )

    def stop_services(
        self,
        request: ExperimentRequest,
        *,
        experiment: ExperimentSpec | None = None,
    ) -> tuple[ManagedServiceStatus, ...]:
        selected = experiment or self.experiment(request.experiment_id)
        selected = _experiment_with_request_overrides(selected, request)
        specs = managed_services_for_systems(
            _selected_request_system_ids(selected, request)
        )
        return stop_managed_services(
            specs,
            repo_root=self.repo_root,
            env=self.env,
        )

    def resolve_run_plan(
        self,
        request: ExperimentRequest,
        *,
        run_id: str,
        experiment: ExperimentSpec | None = None,
        asset_overlay: dict[str, str] | None = None,
    ) -> ResolvedRunPlan:
        selected = experiment or self.experiment(request.experiment_id)
        request = _request_with_selection_lock(selected, request, self.repo_root)
        resolved = _experiment_with_request_overrides(selected, request)
        preset = select_preset(resolved, request.preset)
        workloads = select_workloads(
            resolved, preset, list(request.workloads) or None
        ) or [
            WorkloadSpec(
                id="harbor",
                runner="harbor",
                manifest=request.manifest or resolved.manifest,
            )
        ]
        if asset_overlay is None:
            _require_saved_evaluation_assets(
                self.repo_root, resolved, request, workloads
            )
        jobs = tuple(
            self.rendered_jobs(
                request,
                run_id=run_id,
                write_configs=False,
                experiment=selected,
                asset_overlay=asset_overlay,
            )
        )
        run_name = (
            jobs[0].run_name
            if jobs
            else request.run_name or resolved.run_name or resolved.id
        )
        cells = tuple(
            plan_cells(
                list(jobs),
                run_id=run_id,
                run_name=run_name,
                scheduling_seed=preset.scheduling_seed,
                verify_inputs=False,
            )
        )
        return ResolvedRunPlan(
            request=request,
            experiment=resolved,
            preset=preset,
            workloads=tuple(workloads),
            jobs=jobs,
            cells=cells,
            run_name=run_name,
            max_workers=(
                request.n_concurrent
                or preset.n_concurrent
                or resolved.n_concurrent
                or 2
            ),
        )

    def _materialize_run_plan(
        self, plan: ResolvedRunPlan, *, run_id: str
    ) -> ResolvedRunPlan:
        jobs = tuple(
            self.rendered_jobs(
                plan.request,
                run_id=run_id,
                write_configs=True,
                experiment=plan.experiment,
            )
        )
        cells = tuple(
            plan_cells(
                list(jobs),
                run_id=run_id,
                run_name=plan.run_name,
                scheduling_seed=plan.preset.scheduling_seed,
            )
        )
        if _plan_coordinates(cells) != _plan_coordinates(plan.cells):
            raise RuntimeError(
                "materialized run coordinates differ from the resolved plan"
            )
        return replace(plan, jobs=jobs, cells=cells)

    def prepare_context(
        self,
        request: ExperimentRequest,
        *,
        experiment: ExperimentSpec | None = None,
        rebuild: bool = False,
        run_id: str | None = None,
        _plan: ResolvedRunPlan | None = None,
    ) -> tuple[ContextPreparation, ...]:
        plan = _plan or self.resolve_run_plan(
            request,
            run_id=run_id or "setup-context",
            experiment=experiment,
        )
        request = plan.request
        selected = plan.experiment
        env = self.env
        if "graphiti" in _selected_request_system_ids(selected, request):
            env = managed_service_environment(env, repo_root=self.repo_root)
        env["FUGUE_BUILDER_MODEL"] = (
            request.builder_model
            or selected.builder_model
            or env.get("FUGUE_BUILDER_MODEL")
            or request.model
            or selected.model
            or ""
        )
        runtime = ContextRuntime(
            repo_root=self.repo_root,
            cache_root=self.repo_root / DEFAULT_CACHE_ROOT,
            env=env,
        )
        preset = plan.preset
        workloads = list(plan.workloads)
        for workload in workloads:
            if workload.runner != "harbor":
                continue
            manifest_path = _resolve(
                self.repo_root,
                request.manifest or workload.manifest or selected.manifest,
            )
            materialize_manifest_dataset(
                load_manifest(manifest_path),
                self.repo_root,
                # Dataset paths are content-addressed. --rebuild invalidates
                # prepared context, never an already verified dataset snapshot.
                rebuild=False,
            )
        targets = _preparation_targets(
            experiment=selected,
            workloads=workloads,
            preset=preset,
            requested_systems=(
                list(request.systems)
                or (
                    _request_variant_system_ids(selected, request)
                    if request.variants
                    else None
                )
            ),
            manifest_override=request.manifest,
            repo_root=self.repo_root,
            requested_variants=list(request.variants) or None,
            requested_n_tasks=request.n_tasks,
        )
        records: list[ContextPreparation] = []
        checkouts: dict[tuple[str, str, str], RepositorySnapshot] = {}
        prepared_runtimes: set[str] = set()
        for target in targets:
            snapshot = target.snapshot
            spec = target.spec
            system_id = spec.id
            checks = asyncio.run(preflight_context(spec, runtime, phase="host"))
            failed = [
                check
                for check in checks
                if check.name != "managed runtime"
                if not check.ok and check.severity == "required"
            ]
            if failed:
                records.append(
                    ContextPreparation(
                        system_id,
                        snapshot.task_id,
                        "skipped",
                        "; ".join(f"{item.name}: {item.detail}" for item in failed),
                        variant_id=target.variant_id,
                        config_digest=target.config_digest,
                        retrieval_mode=str(spec.config.get("retrieval_mode") or "")
                        or None,
                    )
                )
                continue
            if (
                runtime_spec(system_id) is not None
                and system_id not in prepared_runtimes
            ):
                ready, _ = runtime_ready(system_id, self.repo_root)
                if rebuild or not ready:
                    prepare_runtime(system_id, repo_root=self.repo_root)
                prepared_runtimes.add(system_id)
            checkout = snapshot
            if system_id != "none" and checkout.checkout == self.repo_root:
                snapshot_key = (snapshot.task_id, snapshot.repo, snapshot.commit)
                checkout = checkouts.get(snapshot_key) or checkout_repository(
                    task_id=snapshot.task_id,
                    repo=snapshot.repo,
                    commit=snapshot.commit,
                    checkout_root=runtime.cache_root / "checkouts",
                    dataset_id=snapshot.dataset_id,
                    rebuild=rebuild,
                )
                checkouts[snapshot_key] = checkout
            prepared = asyncio.run(
                trace_async_operation(
                    "fugue.context.prepare",
                    {
                        "experiment_id": selected.id,
                        "preset_id": preset.id,
                        "run_id": run_id,
                        "context_system_id": system_id,
                        "variant_id": target.variant_id,
                        "context_config_digest": target.config_digest,
                        "context_retrieval_mode": spec.config.get("retrieval_mode"),
                        "task_id": snapshot.task_id,
                        "dataset_id": snapshot.dataset_id,
                        "repository": snapshot.repo,
                        "commit": snapshot.commit,
                    },
                    runtime.env,
                    lambda spec=spec, checkout=checkout: build_context(
                        spec,
                        checkout,
                        runtime,
                        rebuild=rebuild,
                    ),
                    lambda value: {
                        "cache_key": value.cache_key,
                        "cache_hit": value.cache_hit,
                        **value.metrics,
                    },
                )
            )
            records.append(
                ContextPreparation(
                    system_id,
                    snapshot.task_id,
                    "cached" if prepared.cache_hit else "built",
                    prepared.path.as_posix(),
                    cache_key=prepared.cache_key,
                    path=prepared.path,
                    variant_id=target.variant_id,
                    config_digest=target.config_digest,
                    retrieval_mode=str(spec.config.get("retrieval_mode") or "") or None,
                )
            )
        return tuple(records)

    def prepare(
        self,
        request: ExperimentRequest,
        *,
        experiment: ExperimentSpec | None = None,
        rebuild: bool = False,
    ) -> SetupPreparation:
        """Prepare every locked artifact selected by the resolved plan."""
        plan = self.resolve_run_plan(
            request,
            run_id="setup",
            experiment=experiment,
        )
        request = plan.request
        selected = plan.experiment
        preset = plan.preset
        workloads = list(plan.workloads)
        harnesses = list(
            dict.fromkeys(
                job.harness for job in plan.jobs if agent_runtime_spec(job.harness)
            )
        )
        selected_tasks: list[tuple[BenchmarkManifest, Any]] = []
        workload_datasets: list[PreparedWorkloadDataset] = []
        evaluation_asset_locks: set[str] = set()
        dataset_runtime = ContextRuntime(
            repo_root=self.repo_root,
            cache_root=self.repo_root / DEFAULT_CACHE_ROOT,
            env=self.env,
        )
        for workload in workloads:
            if workload.runner != "harbor":
                if workload.dataset:
                    dataset = load_workload_dataset(
                        _resolve(self.repo_root, Path(workload.dataset))
                    )
                    prepared_dataset = prepare_workload_dataset(
                        dataset,
                        dataset_runtime,
                        preset_id=preset.id,
                        rebuild=rebuild,
                    )
                    if prepared_dataset is not None:
                        workload_datasets.append(prepared_dataset)
                continue
            manifest = load_manifest(
                _resolve(
                    self.repo_root,
                    request.manifest or workload.manifest or selected.manifest,
                )
            )
            materialize_manifest_dataset(manifest, self.repo_root, rebuild=False)
            evaluation_assets = prepare_evaluation_assets(manifest, self.repo_root)
            if evaluation_assets is not None:
                evaluation_asset_locks.add(evaluation_assets.as_posix())
            limit = (
                request.n_tasks
                or preset_workload_int(preset, workload.id, "n_tasks")
                or workload.n_tasks
                or preset.n_tasks
            )
            tasks = manifest.tasks[:limit] if limit else manifest.tasks
            selected_tasks.extend((manifest, task) for task in tasks)

        prepared_tasks: list[TaskRuntimePreparation] = []
        seen_tasks: set[tuple[str, str, str]] = set()
        for manifest, task in selected_tasks:
            architecture = task_architecture(task)
            key = (manifest.dataset.harbor_ref, task.id, architecture)
            if key in seen_tasks:
                continue
            seen_tasks.add(key)
            previous_lock = read_task_runtime_lock(manifest, task, self.repo_root)
            gold = load_evaluation_gold_patch(manifest, task.id, self.repo_root)
            lock = prepare_task_runtime(
                manifest,
                task,
                repo_root=self.repo_root,
                rebuild=rebuild,
                gold_patch=gold.patch if gold else None,
            )
            prepared_tasks.append(
                TaskRuntimePreparation(
                    task_id=task.id,
                    architecture=architecture,
                    status=(
                        "cached"
                        if previous_lock is not None
                        and previous_lock.get("recipe_sha256")
                        == lock.get("recipe_sha256")
                        and not rebuild
                        else "built"
                    ),
                    image=str(lock["image"]),
                    image_id=str(lock["image_id"]),
                    recipe_sha256=str(lock["recipe_sha256"]),
                    verification_required=task_runtime_requires_verification(
                        manifest, task
                    ),
                    verification=(
                        dict(lock["verification"])
                        if isinstance(lock.get("verification"), dict)
                        else None
                    ),
                )
            )

        architectures = {task_architecture(task) for _manifest, task in selected_tasks}
        prepared_agents: list[AgentRuntimePreparation] = []
        for harness in dict.fromkeys(harnesses):
            runtime = agent_runtime_spec(harness)
            if runtime is None:
                continue
            for architecture in sorted(architectures & set(runtime.architectures)):
                was_ready, _ = agent_runtime_ready(
                    harness, self.repo_root, architecture
                )
                lock = prepare_agent_runtime(
                    harness,
                    repo_root=self.repo_root,
                    architecture=architecture,
                    rebuild=rebuild,
                )
                prepared_agents.append(
                    AgentRuntimePreparation(
                        harness=harness,
                        architecture=architecture,
                        status="cached" if was_ready and not rebuild else "built",
                        image=str(lock["image"]),
                        image_id=str(lock["image_id"]),
                        recipe_sha256=str(lock["recipe_sha256"]),
                    )
                )
        portable = None
        selected_systems = set(_selected_request_system_ids(selected, request))
        if any(
            variant.enabled
            and variant.context.system_id in selected_systems
            and variant.context.system_id != "none"
            and variant.context.delivery == "portable"
            for variant in selected.variants
        ):
            portable_was_ready, _ = portable_runtime_ready(self.repo_root)
            lock = prepare_portable_runtime(self.repo_root, rebuild=rebuild)
            portable = AgentRuntimePreparation(
                harness="portable-context",
                architecture=str(lock.get("architecture") or "unknown"),
                status=("cached" if portable_was_ready and not rebuild else "built"),
                image=str(lock["image"]),
                image_id=str(lock["image_id"]),
                recipe_sha256=str(lock["recipe_sha256"]),
            )
        return SetupPreparation(
            context=self.prepare_context(
                request,
                experiment=selected,
                rebuild=rebuild,
                _plan=plan,
            ),
            agent_runtimes=tuple(prepared_agents),
            task_runtimes=tuple(prepared_tasks),
            workload_datasets=tuple(workload_datasets),
            evaluation_asset_locks=tuple(sorted(evaluation_asset_locks)),
            portable_context_runtime=portable,
        )

    def prepare_skills(
        self,
        request: ExperimentRequest,
        *,
        experiment: ExperimentSpec | None = None,
        refresh: bool = False,
    ) -> tuple[SkillInspection, ...]:
        """Fetch and inspect selected remote skill directories without executing them."""
        selected = experiment or self.experiment(request.experiment_id)
        selected = _experiment_with_request_overrides(selected, request)
        remote_ids = set(list_skill_source_ids(self.repo_root))
        selected_ids = dict.fromkeys(
            skill_id
            for variant in selected.variants
            if variant.enabled
            for skill_id in variant.skills
            if skill_id in remote_ids
        )
        return tuple(
            prepare_skill_source(skill_id, self.repo_root, refresh=refresh)
            for skill_id in selected_ids
        )

    def approve_skill(
        self,
        skill_id: str,
        digest: str,
        *,
        acknowledged_findings: tuple[str, ...] = (),
    ) -> SkillLockEntry:
        """Approve exactly one inspected skill digest and update the project lock."""
        return approve_skill_source(
            skill_id,
            digest,
            self.repo_root,
            acknowledged_findings=acknowledged_findings,
        )

    def experiment(self, experiment_id: str) -> ExperimentSpec:
        return get_experiment(experiment_id, self.repo_root)

    def experiment_items(self) -> list[tuple[str, str]]:
        return [(item.title, item.id) for item in list_experiments(self.repo_root)]

    def agent_preset_items(self) -> list[tuple[str, str]]:
        return [(item.title, item.id) for item in list_agent_presets(self.repo_root)]

    def apply_agent_preset(
        self, experiment: ExperimentSpec, preset_id: str
    ) -> ExperimentSpec:
        del experiment
        preset = get_agent_preset(preset_id, self.repo_root)
        base = self.experiment(preset.base_experiment_id)
        variant = FeatureVariant(
            id=f"{preset.role}-recommended",
            label=f"Recommended {preset.role}",
            prompt_id=preset.prompt_id,
            skills=list(preset.candidate.skills),
            context=preset.context,
            integrations=list(preset.integrations),
            agent_kwargs=dict(preset.agent_kwargs),
            agent_env=dict(preset.agent_env),
            environment=dict(preset.environment),
            verifier=dict(preset.verifier),
            retry=dict(preset.retry),
            artifacts=list(preset.artifacts),
        )
        return experiment_with_overrides(
            base,
            model=preset.model,
            harnesses=[preset.harness],
            variants=[variant.to_dict()],
            tags=[*base.tags, "agent-preset", f"role:{preset.role}"],
        )

    def request_for_experiment(self, experiment: ExperimentSpec) -> ExperimentRequest:
        """Build UI selections without turning inherited settings into overrides."""
        preset = select_preset(experiment, experiment.default_preset)
        workloads = select_workloads(experiment, preset, None)
        return ExperimentRequest(
            experiment_id=experiment.id,
            preset=preset.id if preset.id != "default" else None,
            workloads=tuple(item.id for item in workloads),
            harnesses=tuple(preset.harnesses or experiment.harnesses),
            variants=tuple(
                variant.id for variant in experiment.variants if variant.enabled
            ),
            run_name=experiment.run_name,
            tags=tuple(experiment.tags),
            jobs_dir=experiment.jobs_dir,
        )

    def save_working_experiment(
        self,
        experiment: ExperimentSpec,
        request: ExperimentRequest,
        *,
        experiment_id: str,
        title: str | None = None,
        assets: tuple[Any, ...] = (),
        replace_assets: bool = False,
    ) -> ExperimentSpec:
        """Persist one explicitly accepted TUI or assistant plan."""
        experiment_id = validate_id(experiment_id, kind="experiment id")
        asset_paths: list[Path] = []
        for asset in assets:
            if asset.kind == "prompt":
                path = self.repo_root / "configs/fugue/prompts" / f"{asset.id}.md"
            elif asset.kind == "skill":
                path = self.repo_root / "configs/fugue/skills" / asset.id / "SKILL.md"
            else:
                path = self.repo_root / evaluation_asset_path(asset.kind, asset.id)
            if path.exists() and not replace_assets:
                raise FileExistsError(
                    f"asset {path.relative_to(self.repo_root)} exists; "
                    "confirm replacement explicitly"
                )
            asset_paths.append(path)

        selected_variants = set(request.variants)
        data = experiment.to_dict()
        data.update(
            {
                "id": experiment_id,
                "title": title or experiment.title,
                "model": request.model or experiment.model,
                "builder_model": request.builder_model or experiment.builder_model,
                "judge_model": request.judge_model or experiment.judge_model,
                "run_name": request.run_name or experiment_id,
                "tags": list(request.tags),
                "harnesses": list(request.harnesses),
                "n_attempts": request.n_attempts or experiment.n_attempts,
                "n_tasks": request.n_tasks or experiment.n_tasks,
                "n_concurrent": request.n_concurrent or experiment.n_concurrent,
                "trace_content": request.trace_content or experiment.trace_content,
                "variants": [
                    {
                        **variant.to_dict(),
                        "enabled": (
                            variant.id in selected_variants
                            if selected_variants
                            else variant.enabled
                        ),
                    }
                    for variant in experiment.variants
                ],
            }
        )
        preset_ids = {item.id for item in experiment.presets}
        if request.preset in preset_ids:
            data["default_preset"] = request.preset
        # Validate the exact experiment and every asset before changing tracked files.
        experiment_from_data(data, item_id=experiment_id)
        staged: list[tuple[Path, Path, bytes | None]] = []
        try:
            for asset, path in zip(assets, asset_paths, strict=True):
                body = str(asset.body)
                if not body.strip():
                    raise ValueError(f"proposed asset is empty: {path}")
                if asset.kind == "evaluation_cases":
                    load_cases(path, text=body)
                elif asset.kind == "evaluation_rubric":
                    load_rubric(path, text=body)
                elif asset.kind == "evaluation_manifest":
                    load_manifest(path, text=body)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(
                    f".{path.name}.{os.getpid()}.{len(staged)}.tmp"
                )
                temporary.write_text(body, encoding="utf-8")
                staged.append(
                    (temporary, path, path.read_bytes() if path.exists() else None)
                )
            for temporary, path, _ in staged:
                os.replace(temporary, path)
            # The experiment is the commit marker and is always written last.
            return save_experiment_data(experiment_id, data, self.repo_root)
        except BaseException:
            for temporary, path, original in reversed(staged):
                temporary.unlink(missing_ok=True)
                if original is None:
                    path.unlink(missing_ok=True)
                    parent = path.parent
                    while parent != self.repo_root and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                else:
                    restore = path.with_name(f".{path.name}.{os.getpid()}.restore")
                    restore.write_bytes(original)
                    os.replace(restore, path)
            raise

    def preview(self, request: ExperimentRequest) -> PreviewSummary:
        return self._preview(request)

    def preview_experiment(
        self,
        experiment: ExperimentSpec,
        *,
        request: ExperimentRequest | None = None,
        asset_overlay: dict[str, str] | None = None,
    ) -> PreviewSummary:
        return self._preview(
            request or _request_for_experiment(experiment),
            experiment=experiment,
            asset_overlay=asset_overlay,
        )

    def _preview(
        self,
        request: ExperimentRequest,
        *,
        experiment: ExperimentSpec | None = None,
        asset_overlay: dict[str, str] | None = None,
    ) -> PreviewSummary:
        plan = self.resolve_run_plan(
            request,
            run_id="preview",
            experiment=experiment,
            asset_overlay=asset_overlay,
        )
        jobs = plan.jobs
        estimated_trials = 0
        for job in jobs:
            if not job.applicable:
                continue
            task_count = int((job.config.get("fugue") or {}).get("task_count") or 1)
            estimated_trials += task_count * job.n_attempts
        return PreviewSummary(
            cells=len(jobs),
            applicable_cells=sum(job.applicable for job in jobs),
            estimated_trials=estimated_trials,
            harnesses=tuple(
                sorted(
                    {
                        job.harness
                        for job in jobs
                        if job.harness not in {"direct", "sequence"}
                    }
                )
            ),
            variants=tuple(sorted({job.variant_id for job in jobs})),
            systems=tuple(sorted({job.context_system_id for job in jobs})),
            workloads=tuple(sorted({job.workload_id for job in jobs})),
            commands=tuple(" ".join(job.command) for job in jobs if job.applicable),
            matrix_cells=tuple(
                PreviewCellSummary(
                    harness=job.harness,
                    variant_id=job.variant_id,
                    variant_label=job.variant_label,
                    context_system_id=job.context_system_id,
                    workload_id=job.workload_id,
                    task_id=job.task_id,
                    trial_count=job.n_attempts
                    * int((job.config.get("fugue") or {}).get("task_count") or 1),
                    applicable=job.applicable,
                    reason=job.skip_reason,
                    context_cache_ready=job.context_cache_ready,
                    context_delivery=job.context_delivery,
                )
                for job in jobs
            ),
        )

    def rendered_jobs(
        self,
        request: ExperimentRequest,
        *,
        run_id: str,
        write_configs: bool = True,
        experiment: ExperimentSpec | None = None,
        asset_overlay: dict[str, str] | None = None,
    ) -> list[RenderedJob]:
        selected = experiment or self.experiment(request.experiment_id)
        request = _request_with_selection_lock(selected, request, self.repo_root)
        selected = _experiment_with_request_overrides(selected, request)
        env = self.env
        if "graphiti" in _selected_request_system_ids(selected, request):
            env = managed_service_environment(
                env,
                repo_root=self.repo_root,
                planning=not write_configs,
            )
        env |= trace_env_defaults(env)
        if request.model:
            env["FUGUE_MODEL"] = request.model
        env["FUGUE_BUILDER_MODEL"] = (
            request.builder_model
            or selected.builder_model
            or env.get("FUGUE_BUILDER_MODEL")
            or request.model
            or selected.model
            or ""
        )
        env["FUGUE_JUDGE_MODEL"] = (
            request.judge_model
            or selected.judge_model
            or env.get("FUGUE_JUDGE_MODEL")
            or ""
        )
        run_name = _run_name(request.run_name or selected.run_name, env)
        env["FUGUE_RUN_NAME"] = run_name
        env["FUGUE_RUN_GROUP"] = env.get("FUGUE_RUN_GROUP", "").strip() or run_name
        preset = select_preset(selected, request.preset)
        workloads = select_workloads(selected, preset, list(request.workloads) or None)
        if not workloads:
            workloads = [
                WorkloadSpec(
                    id="harbor",
                    runner="harbor",
                    manifest=request.manifest or selected.manifest,
                )
            ]
        if write_configs:
            _require_saved_evaluation_assets(
                self.repo_root,
                selected,
                request,
                workloads,
            )
        rendered: list[RenderedJob] = []
        requested_systems = list(request.systems) or (
            _request_variant_system_ids(selected, request) if request.variants else None
        )
        source_provenance = resolve_fugue_source_provenance(self.repo_root)
        for workload in workloads:
            if workload.runner == "harbor":
                manifest_path = _resolve(
                    self.repo_root,
                    request.manifest or workload.manifest or selected.manifest,
                )
                manifest_text = (asset_overlay or {}).get(
                    manifest_path.relative_to(self.repo_root).as_posix()
                    if manifest_path.is_relative_to(self.repo_root)
                    else manifest_path.as_posix()
                )
                manifest = attach_evaluation_assets(
                    load_manifest(manifest_path, text=manifest_text),
                    self.repo_root,
                    required=write_configs,
                )
                env["FUGUE_TAGS"] = ",".join(
                    _run_tags(
                        env=env,
                        tags=request.tags,
                        run_name=run_name,
                        manifest=manifest,
                        manifest_path=manifest_path,
                    )
                )
                renderer = render_jobs if write_configs else preview_jobs
                rendered.extend(
                    renderer(
                        experiment=selected,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        repo_root=self.repo_root,
                        env=env,
                        model=request.model,
                        harness_names=list(request.harnesses) or preset.harnesses,
                        system_names=selected_system_ids(
                            selected,
                            workload,
                            preset,
                            requested_systems,
                        ),
                        variant_names=(
                            list(request.variants) or workload.variants or None
                        ),
                        harness_assignment=workload.harness_assignment,
                        n_tasks=(
                            request.n_tasks
                            or preset_workload_int(preset, workload.id, "n_tasks")
                            or workload.n_tasks
                            or preset.n_tasks
                        ),
                        n_attempts=(
                            request.n_attempts
                            or preset_workload_int(preset, workload.id, "n_attempts")
                            or workload.n_attempts
                            or preset.n_attempts
                        ),
                        n_concurrent=(
                            request.n_concurrent
                            or preset_workload_int(preset, workload.id, "n_concurrent")
                            or preset.n_concurrent
                        ),
                        jobs_dir=request.jobs_dir,
                        run_name=run_name,
                        tags=list(request.tags),
                        run_id=run_id,
                        workload_id=workload.id,
                        preset_id=preset.id if preset.id != "default" else None,
                        required_capabilities=workload.required_capabilities,
                        workload_artifacts=workload.artifacts,
                        scorer_refs=[
                            scorer_reference(item) for item in workload.scorers
                        ],
                        asset_overlay=asset_overlay,
                        source_provenance=source_provenance,
                        scheduling_seed=preset.scheduling_seed,
                    )
                )
            else:
                rendered.extend(
                    self._direct_workload_jobs(
                        experiment=selected,
                        workload=workload,
                        preset=preset,
                        env=env,
                        run_name=run_name,
                        request=request,
                        run_id=run_id,
                        source_provenance=source_provenance,
                    )
                )
        return rendered

    def _direct_workload_jobs(
        self,
        *,
        experiment: ExperimentSpec,
        workload: WorkloadSpec,
        preset: PresetSpec,
        env: dict[str, str],
        run_name: str,
        request: ExperimentRequest,
        run_id: str,
        source_provenance: dict[str, Any],
    ) -> list[RenderedJob]:
        if not workload.dataset:
            raise ValueError(f"workload {workload.id} requires dataset")
        dataset_path = _resolve(self.repo_root, Path(workload.dataset))
        dataset = load_workload_dataset(dataset_path)
        if dataset.runner != workload.runner:
            raise ValueError(
                f"workload {workload.id} runner {workload.runner} does not match "
                f"{dataset.runner} dataset"
            )
        systems = (
            selected_system_ids(
                experiment,
                workload,
                preset,
                list(request.systems)
                or (
                    _request_variant_system_ids(experiment, request)
                    if request.variants
                    else None
                ),
            )
            or []
        )
        selected_model = select_model(
            request.model,
            env=env,
            experiment_model=experiment.model,
        )
        builder_model = env.get("FUGUE_BUILDER_MODEL") or selected_model
        route = resolve_model_route(builder_model, env)
        attempts = (
            request.n_attempts
            or preset_workload_int(preset, workload.id, "n_attempts")
            or workload.n_attempts
            or preset.n_attempts
            or experiment.n_attempts
            or 1
        )
        limit = (
            request.n_tasks
            or preset_workload_int(preset, workload.id, "n_tasks")
            or workload.n_tasks
            or preset.n_tasks
        )
        selected_variants = [
            variant
            for variant in experiment.variants
            if variant.enabled
            and variant.context.system_id in systems
            and (
                variant.id in request.variants
                if request.variants
                else not workload.variants or variant.id in workload.variants
            )
        ]
        jobs: list[RenderedJob] = []
        for variant in selected_variants:
            system_id = variant.context.system_id
            system_env = (
                managed_service_environment(env, repo_root=self.repo_root)
                if system_id == "graphiti"
                else without_managed_service_environment(env)
            )
            direct_env = dict(system_env)
            direct_env["FUGUE_MODEL"] = selected_model
            runtime = ContextRuntime(
                repo_root=self.repo_root,
                cache_root=self.repo_root / DEFAULT_CACHE_ROOT,
                env=system_env,
            )
            base_spec = get_context_system(system_id, self.repo_root)
            spec = replace(
                base_spec,
                config={**base_spec.config, **variant.context.config},
            )
            delivery = variant.context.delivery
            license_env = f"FUGUE_LICENSE_APPROVED_{_env_id(system_id)}"
            license_blocked = spec.requires_license_approval and system_env.get(
                license_env, ""
            ).lower() not in {"1", "true", "yes"}
            resolution = resolve_context_capabilities(
                spec,
                delivery=delivery,
                runner=workload.runner,
                additional=workload.required_capabilities,
            )
            skip_reason = resolution.reason
            if skip_reason is None and license_blocked:
                skip_reason = f"license approval required via {license_env}"
            elif skip_reason is None:
                failed = [
                    check
                    for check in asyncio.run(
                        preflight_context(spec, runtime, phase="host")
                    )
                    if not check.ok and check.severity == "required"
                ]
                if failed:
                    skip_reason = "; ".join(
                        f"{check.name}: {check.detail}" for check in failed
                    )
            command = [
                sys.executable,
                "-m",
                "fugue.bench.cli",
                "_context-evaluate",
                "--experiment",
                experiment.id,
                "--workload",
                workload.id,
                "--system",
                system_id,
                "--variant",
                variant.id,
                "--preset",
                preset.id,
                "--run-id",
                run_id,
                "--attempts",
                str(attempts),
                "--concurrency",
                str(
                    request.n_concurrent
                    or preset.n_concurrent
                    or experiment.n_concurrent
                    or 4
                ),
                "--repo-root",
                self.repo_root.as_posix(),
            ]
            if limit:
                command.extend(["--limit", str(limit)])
            local_count = (
                len(dataset.retrieval_cases)
                if workload.runner == "retrieval"
                else len(dataset.sequence_cases)
            )
            count = int(
                ((dataset.source.get("counts") or {}).get(preset.id)) or local_count
            )
            task_count = min(count, limit) if limit else count
            config = {
                "fugue": {
                    "experiment_id": experiment.id,
                    "preset_id": preset.id,
                    "workload_id": workload.id,
                    "runner": workload.runner,
                    "context_system_id": system_id,
                    "variant_id": variant.id,
                    "context_delivery": delivery,
                    "context_version": spec.version,
                    "dataset": dataset_path.as_posix(),
                    "task_count": task_count,
                    "n_attempts": attempts,
                    "applicable": skip_reason is None,
                    "skip_reason": skip_reason,
                }
            }
            initial_comparison_example_id = comparison_example_id(
                dataset_id=dataset.id,
                workload_id=workload.id,
                logical_task_id=dataset.id,
            )
            direct_harness = "direct" if workload.runner == "retrieval" else "sequence"
            resolved_candidate = resolve_candidate(
                harness=direct_harness,
                harness_version=f"fugue-{workload.runner}@1",
                model_route=model_route_identity(route),
                prompt_digest=None,
                skills=(),
                context={
                    "id": system_id,
                    "version": spec.version,
                    "config": spec.config,
                    "delivery": delivery,
                },
                integrations=(),
                agent={},
                execution={
                    "runner": workload.runner,
                    "n_attempts": attempts,
                    "trace_content": experiment.trace_content,
                    "scheduling_seed": preset.scheduling_seed,
                    "fugue_source": source_provenance,
                },
            )
            candidate_id = resolved_candidate.candidate_id
            jobs.append(
                RenderedJob(
                    command=command,
                    config_path=dataset_path,
                    result_path=self.repo_root
                    / ".fugue"
                    / "runtime"
                    / run_id
                    / "context-results.jsonl",
                    config=config,
                    env={
                        **direct_env,
                        "FUGUE_CANDIDATE_ID": candidate_id,
                        "FUGUE_EXECUTION_FINGERPRINT": resolved_candidate.execution_fingerprint,
                        "FUGUE_EXECUTION_KIND": "provider_diagnostic",
                        "FUGUE_IDENTITY_SCHEMA_VERSION": str(
                            CANDIDATE_IDENTITY_SCHEMA_VERSION
                        ),
                        "FUGUE_DATASET": dataset.id,
                        "FUGUE_WORKLOAD_ID": workload.id,
                        "FUGUE_CONTEXT_SYSTEM_ID": system_id,
                        "FUGUE_CONTEXT_DELIVERY": delivery,
                        "FUGUE_VARIANT_ID": variant.id,
                    },
                    job_name=f"{_slug(run_name)}-{workload.id}-{variant.id}",
                    harness=direct_harness,
                    context_system_id=system_id,
                    context_delivery=delivery,
                    context_version=spec.version,
                    context_cache_keys={},
                    context_cache_ready=False,
                    prompt_id=None,
                    skill_ids=[],
                    variant_id=variant.id,
                    variant_label=variant.label,
                    agent_config_hash="",
                    route=route,
                    workload_id=workload.id,
                    preset_id=preset.id,
                    run_id=run_id,
                    run_name=run_name,
                    task_id=dataset.id,
                    trial_index=1,
                    n_attempts=attempts,
                    comparison_example_id=initial_comparison_example_id,
                    candidate_id=candidate_id,
                    resolved_candidate=resolved_candidate,
                    execution_kind="provider_diagnostic",
                    applicable=skip_reason is None,
                    skip_reason=skip_reason,
                )
            )
        return jobs

    async def compose_experiment(
        self,
        request: str,
        *,
        base_experiment: str | ExperimentSpec = "pilot",
        model: str | None = None,
        trace_content: str | None = None,
    ) -> Any:
        from fugue.bench.ai import ExperimentComposer

        return await ExperimentComposer(self).compose(
            request,
            base_experiment=base_experiment,
            model=model,
            trace_content=trace_content,
        )

    async def plan_analysis(
        self,
        question: str,
        *,
        filters: dict[str, str] | None = None,
        model: str | None = None,
        source: str | None = None,
    ) -> Any:
        from fugue.bench.ai import ExperimentAnalyst

        return await ExperimentAnalyst(self).plan(
            question,
            filters=filters,
            model=model,
            source=source,
        )

    def prepare_analysis(self, spec: AnalysisSpec) -> AnalysisPreview:
        from fugue.bench.ai import ExperimentAnalyst

        return ExperimentAnalyst(self).prepare(spec)

    async def execute_analysis(
        self,
        preview: AnalysisPreview,
        *,
        model: str | None = None,
    ) -> AnalysisResult:
        from fugue.bench.ai import ExperimentAnalyst

        return await ExperimentAnalyst(self).execute(preview, model=model)

    def execute_run(
        self,
        request: ExperimentRequest,
        *,
        run_id: str,
        experiment: ExperimentSpec | None = None,
        cell_runner: Any = None,
        cancellation_event: threading.Event | None = None,
    ) -> RunSummary:
        """Own the complete snapshot-before-execution run transaction."""
        run_dir = self.repo_root / ".fugue" / "runtime" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cancel_event = cancellation_event or threading.Event()
        restore_sigterm = _install_worker_sigterm_handler(cancel_event)
        write_run_manifest(
            self.repo_root,
            run_id,
            {
                "status": "starting",
                "run_name": request.run_name or request.experiment_id,
                "experiment_id": request.experiment_id,
                "trace_content": request.trace_content,
            },
        )
        running = False
        live: LiveEvaluationCoordinator | None = None
        try:
            if cancel_event.is_set():
                raise _RunCancellation
            plan = self.resolve_run_plan(
                request,
                run_id=run_id,
                experiment=experiment,
            )
            request = plan.request
            resolved = plan.experiment
            snapshot_path = run_dir / "experiment.yaml"
            temporary = snapshot_path.with_name(
                f".{snapshot_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(experiment_to_yaml(resolved), encoding="utf-8")
            os.replace(temporary, snapshot_path)
            plan = self._materialize_run_plan(plan, run_id=run_id)
            rendered = list(plan.jobs)
            _verify_rendered_setup(rendered)
            validate_harbor_job_configs(
                [
                    job.config_path
                    for job in rendered
                    if job.applicable and job.execution_kind == "agent"
                ]
            )
            run_name = plan.run_name
            selected_preset = plan.preset
            max_workers = plan.max_workers
            cells = list(plan.cells)
            bridge_runtime = _resolved_bridge_runtime(
                rendered,
                request=request,
                experiment=resolved,
                repo_root=self.repo_root,
                env=self.env,
            )
            evaluation_assets = build_evaluation_asset_lock(run_id, cells)
            write_evaluation_asset_lock(self.repo_root, evaluation_assets)
            cells = [
                replace(
                    cell,
                    evaluation_asset_lock_sha256=evaluation_assets.lock_sha256,
                )
                for cell in cells
            ]
            treatment_selection_sha256 = _selection_lock_digest(
                request.selection_lock, self.repo_root
            )
            run_snapshot = build_run_snapshot(
                repo_root=self.repo_root,
                run_id=run_id,
                experiment=resolved,
                request=asdict(request),
                jobs=rendered,
                cells=cells,
                env=self.env,
                bridge_runtime=bridge_runtime,
                evaluation_asset_lock_sha256=evaluation_assets.lock_sha256,
                treatment_selection_sha256=treatment_selection_sha256,
            )
            source_commit = str(
                (run_snapshot.runtime.get("fugue_source") or {}).get("commit") or ""
            )
            cells = [
                replace(
                    cell,
                    run_snapshot_sha256=run_snapshot.snapshot_sha256,
                    source_commit=source_commit,
                )
                for cell in cells
            ]
            write_run_input_lock(self.repo_root, run_snapshot)
            if cancel_event.is_set():
                raise _RunCancellation
            job_dirs = sorted(
                {
                    str(job.config.get("jobs_dir"))
                    for job in rendered
                    if job.config.get("jobs_dir")
                }
            )
            job_paths = sorted(
                {
                    cell.result_path.parent.as_posix()
                    for cell in cells
                    if cell.applicable
                }
            )
            write_run_manifest(
                self.repo_root,
                run_id,
                {
                    "status": "running",
                    "started_at": _now(),
                    "run_name": run_name,
                    "experiment_id": resolved.id,
                    "trace_project": trace_project_slug(
                        rendered[0].env if rendered else self.env
                    ),
                    "cell_count": len(cells),
                    "jobs_dirs": job_dirs,
                    "job_paths": job_paths,
                    "input_lock": "input-lock.json",
                    "snapshot_sha256": run_snapshot.snapshot_sha256,
                    "scheduling_seed": selected_preset.scheduling_seed,
                    "max_workers": max_workers,
                },
            )
            running = True
            run_env = rendered[0].env if rendered else self.env
            observability_error = None
            if (
                cells
                and run_env.get("WANDB_API_KEY", "").strip()
                and run_env.get("FUGUE_DISABLE_LIVE_EVALUATIONS", "").lower()
                not in {"1", "true", "yes"}
            ):
                try:
                    live = LiveEvaluationCoordinator(
                        cells,
                        repo_root=self.repo_root,
                        project=trace_project_slug(run_env),
                        env=run_env,
                        cancellation_event=cancel_event,
                    )
                except Exception as exc:
                    observability_error = f"{type(exc).__name__}: {exc}"
            local = (
                GeneratedEvaluationCoordinator(
                    cells, repo_root=self.repo_root, env=run_env
                )
                if live is None
                and any(cell.evaluation_case is not None for cell in cells)
                else None
            )
            outcomes = execute_cells(
                cells,
                repo_root=self.repo_root,
                max_workers=max_workers,
                runner=cell_runner,
                cell_started=live.begin_cell if live is not None else None,
                cell_finished=(
                    live.finish_cell
                    if live is not None
                    else local.finish_cell
                    if local is not None
                    else None
                ),
                cancellation_event=cancel_event,
            )
            cancelled = cancel_event.is_set() or any(
                item.status == "cancelled" for item in outcomes
            )
            publication = (
                live.finalize(cancelled=True)
                if live is not None and cancelled
                else live.finalize()
                if live is not None
                else None
            )
            failures = list(publication.failures if publication else ())
            if observability_error:
                failures.insert(0, observability_error)
            failed = sum(item.status == "failed" for item in outcomes)
            skipped = sum(item.status == "not_applicable" for item in outcomes)
            cancelled_cells = sum(item.status == "cancelled" for item in outcomes)
            passed = sum(item.status == "passed" for item in outcomes)
            _finalize_run(
                self.repo_root,
                run_id,
                run_dir=run_dir,
                status=("cancelled" if cancelled else "failed" if failed else "passed"),
                error="Run cancelled by the operator." if cancelled else None,
                running=running,
                values={
                    "passed_cells": passed,
                    "failed_cells": failed,
                    "cancelled_cells": cancelled_cells,
                    "not_applicable_cells": skipped,
                    "observability_status": "failed"
                    if failures
                    else "cancelled"
                    if cancelled
                    else "passed",
                    "evaluation_runs": [
                        asdict(item) for item in publication.evaluations
                    ]
                    if publication
                    else [],
                    "evaluation_failures": failures,
                },
            )
        except _RunCancellation:
            _cancel_live_evaluation(live, "Run cancelled by the operator.")
            _finalize_run(
                self.repo_root,
                run_id,
                run_dir=run_dir,
                status="cancelled",
                error="Run cancelled by the operator.",
                running=running,
                values={
                    "observability_status": "cancelled",
                },
            )
        except KeyboardInterrupt:
            _finalize_run(
                self.repo_root,
                run_id,
                run_dir=run_dir,
                status="interrupted",
                error="Run interrupted",
                running=running,
            )
        except Exception as exc:
            if cancel_event.is_set():
                _cancel_live_evaluation(live, "Run cancelled by the operator.")
                _finalize_run(
                    self.repo_root,
                    run_id,
                    run_dir=run_dir,
                    status="cancelled",
                    error="Run cancelled by the operator.",
                    running=running,
                    values={
                        "observability_status": "cancelled",
                    },
                )
            else:
                error = f"{type(exc).__name__}: {exc}"
                _finalize_run(
                    self.repo_root,
                    run_id,
                    run_dir=run_dir,
                    status="failed",
                    error=error,
                    running=running,
                    values={
                        "phase": "running" if running else "starting",
                    },
                )
                raise
        finally:
            restore_sigterm()
        return self.run_summary(run_id)

    def launch(
        self,
        request: ExperimentRequest,
        *,
        experiment: ExperimentSpec | None = None,
        run_id: str | None = None,
    ) -> RunSummary:
        run_id = validate_id(run_id or new_run_id(), kind="run id")
        selected = experiment or self.experiment(request.experiment_id)
        resolved = _experiment_with_request_overrides(selected, request)
        run_dir = self.repo_root / ".fugue" / "runtime" / run_id
        if (run_dir / "run.json").exists():
            raise ValueError(f"run id already exists: {run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot = run_dir / "experiment.yaml"
        temp = snapshot.with_name(f".{snapshot.name}.{os.getpid()}.tmp")
        temp.write_text(experiment_to_yaml(resolved))
        os.replace(temp, snapshot)
        run_name = request.run_name or selected.run_name or selected.id
        command = [
            sys.executable,
            "-m",
            "fugue.bench.cli",
            *self.request_arguments(request, experiment_file=snapshot),
            "--run-id",
            run_id,
        ]
        env = self.env
        existing_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            self.repo_root.as_posix()
            if not existing_path
            else f"{self.repo_root}{os.pathsep}{existing_path}"
        )
        self.supervisor.start_detached(
            run_id=run_id,
            command=command,
            env=env,
            run_name=run_name,
            experiment_id=selected.id,
        )
        return self.run_summary(run_id)

    def request_arguments(
        self,
        request: ExperimentRequest,
        *,
        experiment_file: Path | None = None,
    ) -> list[str]:
        args = ["run"]
        if experiment_file:
            args.extend(["--experiment-file", experiment_file.as_posix()])
        else:
            args.append(request.experiment_id)
        for flag, value in (
            ("--manifest", request.manifest.as_posix() if request.manifest else None),
            ("--preset", request.preset),
            ("--workloads", _joined(request.workloads)),
            ("--harnesses", _joined(request.harnesses)),
            ("--systems", _joined(request.systems)),
            ("--variants", _joined(request.variants)),
            ("--model", request.model),
            ("--builder-model", request.builder_model),
            ("--judge-model", request.judge_model),
            ("--run-name", request.run_name),
            ("--tags", _joined(request.tags)),
            ("--jobs-dir", request.jobs_dir.as_posix() if request.jobs_dir else None),
            ("--trace-content", request.trace_content),
        ):
            if value:
                args.extend([flag, str(value)])
        for flag, value in (
            ("--n-attempts", request.n_attempts),
            ("--n-tasks", request.n_tasks),
            ("--n-concurrent", request.n_concurrent),
        ):
            if value is not None:
                args.extend([flag, str(value)])
        args.extend(["--repo-root", self.repo_root.as_posix()])
        args.extend(["--env-file", self.env_file.as_posix()])
        return args

    def runs(self, *, recover: bool = True) -> list[RunSummary]:
        return [
            self._summarize_run(run) for run in self.supervisor.list(recover=recover)
        ]

    def run_summary(self, run_id: str, *, recover: bool = True) -> RunSummary:
        return self._summarize_run(self.supervisor.get(run_id, recover=recover))

    def package_candidate(
        self,
        run_id: str,
        candidate_id: str,
        *,
        workspace: Path,
        image: str,
        platform: str = "linux/amd64",
        allow_failed: bool = False,
    ) -> Any:
        from fugue.bench.deployment import package_candidate

        resolved_id = self.resolve_candidate_id(run_id, candidate_id)
        return package_candidate(
            repo_root=self.repo_root,
            run_id=run_id,
            candidate_id=resolved_id,
            workspace=workspace,
            image=image,
            platform=platform,
            allow_failed=allow_failed,
            env=self.env,
        )

    def resolve_candidate_id(self, run_id: str, value: str) -> str:
        candidates = [item.candidate_id for item in self.run_summary(run_id).candidates]
        if value in candidates:
            return value
        matches = [
            candidate_id
            for candidate_id in candidates
            if candidate_id.startswith(value)
        ]
        if not matches:
            raise ValueError(f"candidate prefix does not match this run: {value}")
        if len(matches) > 1:
            raise ValueError(f"candidate prefix is ambiguous: {value}")
        return matches[0]

    def wait_for_run(self, run_id: str, *, poll_sec: float = 0.25) -> RunSummary:
        terminal = {"passed", "failed", "cancelled", "interrupted"}
        while True:
            run = self.run_summary(run_id)
            if run.status in terminal:
                return run
            time.sleep(poll_sec)

    def run_trace_refs(
        self, run_id: str, *, cell_id: str | None = None
    ) -> tuple[AgentTraceRef, ...]:
        run = self.supervisor.get(run_id, recover=False)
        sources = _run_job_paths(self.repo_root, run)
        rows = [
            row
            for row in export_rows([*sources, run.run_dir])
            if row.get("record_type") == "trial" and row.get("run_id") == run_id
        ]
        if cell_id:
            cell = next(
                (
                    item
                    for item in self.run_summary(run_id).cells
                    if item.cell_id == cell_id
                ),
                None,
            )
            if cell is None:
                raise ValueError(f"cell not found in {run_id}: {cell_id}")
            rows = [row for row in rows if _row_matches_cell(row, cell)]
        return _agent_trace_refs(rows)

    def export_run(
        self,
        run_id: str,
        *,
        out: Path | None = None,
        fetch_weave: bool = False,
        to_weave: bool = False,
        republish: bool = False,
        republish_reason: str | None = None,
    ) -> ExportSummary:
        run = self.supervisor.get(run_id, recover=False)
        project = str(run.metadata.get("trace_project") or trace_project_slug(self.env))
        job_paths = _run_job_paths(self.repo_root, run)
        bundle = compile_export(
            [*job_paths, run.run_dir],
            fetch_weave=fetch_weave,
            project=project,
            publish=to_weave,
            ledger_root=self.repo_root / ".fugue" / "runtime" / "publications",
            republish=republish,
            republish_reason=republish_reason,
            env=self.env,
            repo_root=self.repo_root,
        )
        predictions = list(bundle.predictions)
        measurements = list(bundle.measurements)
        output = out or self.repo_root / "reports" / f"{run_id}.jsonl"
        write_jsonl(predictions, output, env=self.env)
        measurement_output = (
            output.with_name(f"{output.stem}.measurements.jsonl")
            if measurements
            else None
        )
        if measurement_output is not None:
            write_jsonl(measurements, measurement_output, env=self.env)
        published = 0
        skipped = 0
        evaluations: tuple[PublishedEvaluation, ...] = ()
        publication_failures: tuple[str, ...] = ()
        if to_weave:
            publication = bundle.publication
            published = publication.published
            skipped = publication.skipped
            evaluations = publication.evaluations
            publication_failures = publication.failures
            direct_evaluations = tuple(
                item for item in evaluations if item.direct_predictions > 0
            )
            if direct_evaluations or publication_failures:
                update_run_manifest(
                    self.repo_root,
                    run_id,
                    lambda manifest: _publication_manifest_update(
                        manifest,
                        direct_evaluations,
                        publication_failures,
                    ),
                )
        return ExportSummary(
            path=output,
            rows=len(predictions),
            measurement_path=measurement_output,
            measurements=len(measurements),
            published=published,
            skipped=skipped,
            evaluations=evaluations,
            publication_failures=publication_failures,
        )

    def results(self, paths: list[Path] | None = None) -> ResultSummary:
        sources = paths or [
            self.repo_root / "jobs",
            self.repo_root / ".fugue" / "runtime",
        ]
        rows = export_rows([path for path in sources if path.exists()])
        if paths is None:
            rows = _merge_report_rows(rows, self.repo_root / "reports")
        trials = normalize_prediction_rows(rows)
        scored = [row for row in trials if row.get("pass") is not None]
        rewards = [_float(row.get("reward")) for row in trials]
        rewards = [value for value in rewards if value is not None]
        times = [_float(row.get("wall_time_sec")) for row in trials]
        times = [value for value in times if value is not None]
        passed = sum(row.get("pass") is True for row in trials)
        return ResultSummary(
            total=len(trials),
            passed=passed,
            failed=sum(row.get("pass") is False for row in trials),
            pass_rate=passed / len(scored) if scored else None,
            cost_usd=sum(float(row.get("cost_usd") or 0) for row in trials),
            input_tokens=sum(int(row.get("n_input_tokens") or 0) for row in trials),
            output_tokens=sum(int(row.get("n_output_tokens") or 0) for row in trials),
            average_reward=sum(rewards) / len(rewards) if rewards else None,
            average_wall_time_sec=sum(times) / len(times) if times else None,
            tool_calls=sum(
                int(row.get("weave_tool_call_count") or 0) for row in trials
            ),
            turns=sum(int(row.get("weave_turn_count") or 0) for row in trials),
            context_assigned=sum(bool(row.get("context_assigned")) for row in trials),
            context_invoked=sum(bool(row.get("context_invoked")) for row in trials),
            context_registered=sum(
                row.get("context_registered") is True for row in trials
            ),
            runtime_mismatched=sum(
                row.get("runtime_equivalence_status") == "mismatch"
                or row.get("runtime_drift") is True
                for row in trials
            ),
            attributed_errors=sum(
                int(row.get(f"{origin}_error_count") or 0)
                for row in trials
                for origin in (
                    "agent",
                    "benchmark_runtime",
                    "harness_adapter",
                    "context_system",
                    "provider",
                    "fugue",
                )
            ),
            linked_traces=sum(
                row.get("trace_link_status") == "linked" for row in trials
            ),
            unlinked_traces=sum(
                row.get("trace_link_status") not in {None, "linked"} for row in trials
            ),
            usage_unavailable=sum(
                row.get("weave_usage_status") == "unavailable" for row in trials
            ),
            rows=tuple(trials),
            agent_traces=_agent_trace_refs(trials),
        )

    def deep_links(
        self,
        *,
        project: str | None = None,
        trace_url: str | None = None,
    ) -> DeepLinks:
        env = self.env
        selected_project = project or trace_project_slug(env)
        base = env.get("WANDB_APP_BASE_URL", "https://wandb.ai").rstrip("/")
        weave = f"{base}/{selected_project}/weave"
        return DeepLinks(
            project=f"{base}/{selected_project}",
            weave=weave,
            agents=f"{weave}/agents",
            trace=trace_url,
        )

    def run_links(self, run_id: str) -> DeepLinks:
        run = self.supervisor.get(run_id, recover=False)
        project = str(run.metadata.get("trace_project") or "").strip() or None
        return self.deep_links(project=project)

    def run_evaluation(
        self, run_id: str, *, cell_id: str | None = None
    ) -> PublishedEvaluation | None:
        run = self.run_summary(run_id, recover=False)
        candidate_id = None
        if cell_id:
            cell = next((item for item in run.cells if item.cell_id == cell_id), None)
            if cell is None:
                raise ValueError(f"cell not found in {run_id}: {cell_id}")
            candidate_id = cell.candidate_id
        return next(
            (
                item
                for item in run.evaluations
                if item.url
                and (candidate_id is None or item.candidate_id == candidate_id)
            ),
            None,
        )

    @staticmethod
    def _role_status(role: str, model: str, env: dict[str, str]) -> ModelRoleStatus:
        route = resolve_model_route(model, env)
        return ModelRoleStatus(
            role=role,
            model=route.display_model,
            provider=route.provider,
            key_env=route.api_key_env,
            key_present=bool(env.get(route.api_key_env, "").strip()),
        )

    def _summarize_run(self, run: ManagedRun) -> RunSummary:
        records = latest_cell_records(run.run_dir / "cells.jsonl")
        cells = tuple(
            CellSummary(
                cell_id=str(item.get("cell_id") or ""),
                candidate_id=str(item.get("candidate_id") or ""),
                status=str(item.get("status") or "unknown"),
                harness=str(item.get("harness") or "direct"),
                variant_id=str(item.get("variant_id") or "baseline"),
                context_system_id=str(item.get("context_system_id") or "none"),
                workload_id=str(item.get("workload_id") or "harbor"),
                task_id=str(item.get("task_id") or ""),
                wall_time_sec=(
                    float(item["wall_time_sec"])
                    if item.get("wall_time_sec") is not None
                    else None
                ),
                error=item.get("error"),
                skip_reason=item.get("skip_reason"),
                context_delivery=str(item.get("context_delivery") or "portable"),
                benchmark_outcome=str(item.get("benchmark_outcome") or "unscored"),
                reward=(
                    float(item["reward"]) if item.get("reward") is not None else None
                ),
            )
            for item in records
        )
        candidates = _candidate_summaries(run.run_dir, cells, records)
        evaluations = tuple(
            PublishedEvaluation(
                candidate_id=str(item.get("candidate_id") or ""),
                name=str(item.get("name") or "evaluation"),
                examples=int(item.get("examples") or 0),
                url=str(item["url"]) if item.get("url") else None,
                evaluation_ref=(
                    str(item["evaluation_ref"]) if item.get("evaluation_ref") else None
                ),
                dataset_ref=(
                    str(item["dataset_ref"]) if item.get("dataset_ref") else None
                ),
                model_ref=str(item["model_ref"]) if item.get("model_ref") else None,
                agent_predictions=int(
                    item.get("agent_predictions")
                    or max(
                        int(item.get("examples") or 0)
                        - int(item.get("direct_predictions") or 0),
                        0,
                    )
                ),
                linked_agent_predictions=int(
                    item.get("linked_agent_predictions")
                    or item.get("linked_predictions")
                    or 0
                ),
                direct_predictions=int(item.get("direct_predictions") or 0),
                linking_failures=tuple(
                    str(value) for value in item.get("linking_failures", []) if value
                ),
                publication_id=(
                    str(item["publication_id"]) if item.get("publication_id") else None
                ),
                revision=int(item.get("revision") or 1),
                supersedes=(
                    str(item["supersedes"]) if item.get("supersedes") else None
                ),
                active=item.get("active") is not False,
            )
            for item in run.metadata.get("evaluation_runs", [])
            if isinstance(item, dict) and item.get("active") is not False
        )
        return RunSummary(
            run_id=run.run_id,
            run_name=run.run_name,
            experiment_id=run.experiment_id,
            status=run.status,
            created_at=run.created_at,
            cells=cells,
            passed=sum(cell.status == "passed" for cell in cells),
            failed=sum(cell.status == "failed" for cell in cells),
            cancelled=sum(cell.status == "cancelled" for cell in cells),
            interrupted=sum(cell.status == "interrupted" for cell in cells),
            pending=sum(cell.status in {"pending", "running"} for cell in cells),
            not_applicable=sum(cell.status == "not_applicable" for cell in cells),
            candidates=candidates,
            log_path=run.log_path,
            observability_status=run.metadata.get("observability_status"),
            evaluations=evaluations,
            evaluation_failures=tuple(
                str(value)
                for value in run.metadata.get("evaluation_failures", [])
                if value
            ),
            cancellation_cleanup_status=run.metadata.get("cancellation_cleanup_status"),
            cancellation_cleanup_projects=tuple(
                str(value)
                for value in run.metadata.get("cancellation_cleanup_projects", [])
                if value
            ),
            cancellation_cleanup_errors=tuple(
                str(value)
                for value in run.metadata.get("cancellation_cleanup_errors", [])
                if value
            ),
        )


def _candidate_summaries(
    run_dir: Path,
    cells: tuple[CellSummary, ...],
    records: list[dict[str, Any]],
) -> tuple[CandidateSummary, ...]:
    from fugue.bench.deployment import candidate_packageability

    lock_path = run_dir / "input-lock.json"
    try:
        snapshot = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        snapshot = {"planned_matrix": []}
    candidate_ids = sorted(
        {
            *(cell.candidate_id for cell in cells if cell.candidate_id),
            *[str(value) for value in (snapshot.get("candidates") or {})],
        }
    )
    prefixes = _unique_candidate_prefixes(candidate_ids)
    definitions = snapshot.get("candidates") or {}
    result: list[CandidateSummary] = []
    for candidate_id in candidate_ids:
        selected = [cell for cell in cells if cell.candidate_id == candidate_id]
        passed = sum(cell.benchmark_outcome == "passed" for cell in selected)
        failed = sum(cell.benchmark_outcome == "failed" for cell in selected)
        execution_failed = sum(cell.status == "failed" for cell in selected)
        cancelled = sum(cell.status == "cancelled" for cell in selected)
        interrupted = sum(cell.status == "interrupted" for cell in selected)
        unscored = sum(
            cell.status == "passed" and cell.benchmark_outcome == "unscored"
            for cell in selected
        )
        pending = sum(cell.status in {"pending", "running"} for cell in selected)
        not_applicable = sum(cell.status == "not_applicable" for cell in selected)
        planned = [
            item
            for item in snapshot.get("planned_matrix") or []
            if item.get("candidate_id") == candidate_id
            and item.get("applicable") is not False
        ]
        terminal = sum(
            cell.status in {"passed", "failed", "cancelled", "interrupted"}
            for cell in selected
        )
        completeness = terminal / len(planned) if planned else 0.0
        packageable, reason = candidate_packageability(
            snapshot,
            [item for item in records if item.get("candidate_id") == candidate_id],
            candidate_id,
        )
        definition = definitions.get(candidate_id) or {}
        model_route = definition.get("model_route") or {}
        context = definition.get("context") or {}
        result.append(
            CandidateSummary(
                candidate_id=candidate_id,
                display_id=prefixes[candidate_id],
                configuration={
                    "harness": definition.get("harness"),
                    "model": model_route.get("display_model")
                    or model_route.get("model_id"),
                    "context": context,
                    "skills": definition.get("skills") or [],
                    "integrations": definition.get("integrations") or [],
                },
                passed=passed,
                failed=failed,
                execution_failed=execution_failed,
                cancelled=cancelled,
                interrupted=interrupted,
                unscored=unscored,
                pending=pending,
                not_applicable=not_applicable,
                completeness=completeness,
                packageable=packageable,
                packageability_reason=reason,
            )
        )
    return tuple(result)


def _unique_candidate_prefixes(candidate_ids: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate_id in candidate_ids:
        length = 12
        while any(
            other != candidate_id and other.startswith(candidate_id[:length])
            for other in candidate_ids
        ):
            length += 1
        result[candidate_id] = candidate_id[:length]
    return result


def select_preset(experiment: ExperimentSpec, requested: str | None) -> PresetSpec:
    if not experiment.presets:
        return PresetSpec(id="default")
    preset_id = requested or experiment.default_preset or experiment.presets[0].id
    for preset in experiment.presets:
        if preset.id == preset_id:
            return preset
    raise ValueError(f"unknown preset: {preset_id}")


def select_workloads(
    experiment: ExperimentSpec,
    preset: PresetSpec,
    requested: list[str] | None,
) -> list[WorkloadSpec]:
    selected = set(
        requested or preset.workloads or [item.id for item in experiment.workloads]
    )
    workloads = [item for item in experiment.workloads if item.id in selected]
    missing = sorted(selected - {item.id for item in workloads})
    if missing:
        raise ValueError(f"unknown workload(s): {', '.join(missing)}")
    return workloads


def preset_workload_int(
    preset: PresetSpec,
    workload_id: str,
    key: str,
) -> int | None:
    value = (preset.workload_overrides.get(workload_id) or {}).get(key)
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(
            f"preset {preset.id} workload {workload_id} {key} must be positive"
        )
    return parsed


def selected_system_ids(
    experiment: ExperimentSpec,
    workload: WorkloadSpec,
    preset: PresetSpec,
    requested: list[str] | None,
) -> list[str] | None:
    if requested:
        return list(dict.fromkeys(requested))
    selected = preset.systems or workload.systems
    if selected:
        allowed = set(workload.systems) if workload.systems else None
        return [item for item in selected if allowed is None or item in allowed]
    values = [
        variant.context.system_id for variant in experiment.variants if variant.enabled
    ]
    return list(dict.fromkeys(values)) or None


def _selected_request_system_ids(
    experiment: ExperimentSpec,
    request: ExperimentRequest,
) -> tuple[str, ...]:
    preset = select_preset(experiment, request.preset)
    workloads = select_workloads(
        experiment,
        preset,
        list(request.workloads) or None,
    ) or [
        WorkloadSpec(
            id="harbor",
            runner="harbor",
            manifest=request.manifest or experiment.manifest,
        )
    ]
    requested = list(request.systems) or (
        _request_variant_system_ids(experiment, request) if request.variants else None
    )
    values: list[str] = []
    for workload in workloads:
        values.extend(
            selected_system_ids(experiment, workload, preset, requested) or []
        )
    return tuple(dict.fromkeys(values))


def _request_variant_system_ids(
    experiment: ExperimentSpec, request: ExperimentRequest
) -> list[str]:
    requested = set(request.variants)
    return list(
        dict.fromkeys(
            variant.context.system_id
            for variant in experiment.variants
            if variant.id in requested
        )
    )


def _request_for_experiment(experiment: ExperimentSpec) -> ExperimentRequest:
    return ExperimentRequest(
        experiment_id=experiment.id,
        preset=experiment.default_preset,
        harnesses=tuple(experiment.harnesses),
        variants=tuple(
            variant.id for variant in experiment.variants if variant.enabled
        ),
        model=experiment.model,
        builder_model=experiment.builder_model,
        judge_model=experiment.judge_model,
        n_attempts=experiment.n_attempts,
        n_tasks=experiment.n_tasks,
        n_concurrent=experiment.n_concurrent,
        run_name=experiment.run_name,
        tags=tuple(experiment.tags),
        jobs_dir=experiment.jobs_dir,
        trace_content=experiment.trace_content,
    )


def _resolved_bridge_runtime(
    jobs: list[RenderedJob],
    *,
    request: ExperimentRequest,
    experiment: ExperimentSpec,
    repo_root: Path,
    env: dict[str, str],
) -> dict[str, Any] | None:
    bridged = [
        job
        for job in jobs
        if job.applicable
        and job.execution_kind == "agent"
        and resolve_harness_model_route(job.route, job.harness)["bridge_required"]
    ]
    if not bridged:
        return None
    target_models = {job.route.display_model for job in bridged}
    if len(target_models) != 1:
        raise ValueError(
            "one immutable run cannot share a bridge across multiple target models"
        )
    target_route = bridged[0].route
    builder_model = (
        request.builder_model
        or experiment.builder_model
        or env.get("FUGUE_BUILDER_MODEL")
        or target_route.display_model
    )
    judge_model = request.judge_model or experiment.judge_model
    return bridge_runtime_attestation(
        target_route,
        repo_root=repo_root,
        builder_route=resolve_model_route(builder_model, env),
        judge_route=(resolve_model_route(judge_model, env) if judge_model else None),
        env=env,
    )


def _verify_rendered_setup(jobs: list[RenderedJob]) -> None:
    missing: list[str] = []
    for job in jobs:
        if not job.applicable or job.execution_kind != "agent":
            continue
        fugue = job.config.get("fugue") or {}
        if job.context_system_id != "none" and not job.context_cache_ready:
            missing.append(
                f"{job.job_name}: context artifact {job.context_system_id} is missing"
            )
        if (
            job.context_delivery == "native_mcp"
            and runtime_spec(job.context_system_id) is not None
            and not fugue.get("context_runtime")
        ):
            missing.append(
                f"{job.job_name}: managed runtime {job.context_system_id} is missing"
            )
        context_runtime = fugue.get("context_runtime") or {}
        if fugue.get("context_runtime_required") is True and not context_runtime.get(
            "image_id"
        ):
            missing.append(f"{job.job_name}: portable context runtime image is missing")
        if agent_runtime_spec(job.harness) is not None and not fugue.get(
            "agent_runtime"
        ):
            missing.append(
                f"{job.job_name}: prepared agent runtime {job.harness} is missing"
            )
        if not fugue.get("task_runtime"):
            missing.append(
                f"{job.job_name}: prepared task image {job.task_id} is missing"
            )
    if missing:
        detail = "\n".join(f"- {item}" for item in missing[:20])
        raise RuntimeError(
            "run setup is incomplete; execute `fugue setup --prepare` before "
            f"starting an immutable run:\n{detail}"
        )


def _preparation_targets(
    *,
    experiment: ExperimentSpec,
    workloads: list[WorkloadSpec],
    preset: PresetSpec,
    requested_systems: list[str] | None,
    manifest_override: Path | None,
    repo_root: Path,
    requested_variants: list[str] | None = None,
    requested_n_tasks: int | None = None,
) -> list[ContextPreparationTarget]:
    targets: dict[tuple[str, str, str, str, str], ContextPreparationTarget] = {}
    selected = workloads or [
        WorkloadSpec(
            id="harbor",
            runner="harbor",
            manifest=manifest_override or experiment.manifest,
        )
    ]
    for workload in selected:
        system_ids = (
            selected_system_ids(
                experiment,
                workload,
                preset,
                requested_systems,
            )
            or []
        )
        variants = [
            variant
            for variant in experiment.variants
            if variant.enabled and variant.context.system_id in system_ids
        ]
        effective: list[tuple[FeatureVariant, ContextSystemSpec]] = []
        for variant in variants:
            base = get_context_system(variant.context.system_id, repo_root)
            spec = replace(
                base,
                config={**base.config, **variant.context.config},
            )
            if resolve_context_capabilities(
                spec,
                delivery=variant.context.delivery,
                runner=workload.runner,
                additional=workload.required_capabilities,
            ).applicable:
                effective.append((variant, spec))
        limit = (
            requested_n_tasks
            or preset_workload_int(preset, workload.id, "n_tasks")
            or workload.n_tasks
            or preset.n_tasks
        )
        snapshots: list[RepositorySnapshot] = []
        if workload.runner == "harbor":
            path = _resolve(
                repo_root,
                manifest_override or workload.manifest or experiment.manifest,
            )
            manifest = load_manifest(path)
            tasks = manifest.tasks[:limit] if limit else manifest.tasks
            snapshots.extend(
                RepositorySnapshot(
                    task.id,
                    task.repo,
                    task.base_commit,
                    _fixture_repository_path(task.repository, repo_root),
                    manifest.dataset.harbor_ref,
                    task.metadata,
                    (
                        task.repository.sha256
                        if isinstance(task.repository, FixtureRepositorySpec)
                        else None
                    ),
                )
                for task in tasks
                if task.repo and task.base_commit
            )
        elif workload.dataset:
            dataset = load_workload_dataset(_resolve(repo_root, Path(workload.dataset)))
            cases = [*dataset.retrieval_cases, *dataset.sequence_cases]
            if limit:
                cases = cases[:limit]
            snapshots.extend(
                RepositorySnapshot(
                    case.id,
                    case.repo,
                    case.commit,
                    repo_root,
                    dataset.id,
                )
                for case in cases
            )
        for variant, spec in effective:
            if requested_variants:
                if variant.id not in requested_variants:
                    continue
            elif workload.variants and variant.id not in workload.variants:
                continue
            for snapshot in snapshots:
                config_digest = hashlib.sha256(
                    json.dumps(
                        variant.context.config,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                key = (
                    variant.context.system_id,
                    f"{variant.context.delivery}:{config_digest}",
                    snapshot.task_id,
                    snapshot.repo,
                    snapshot.commit,
                )
                targets[key] = ContextPreparationTarget(
                    variant_id=variant.id,
                    spec=spec,
                    delivery=variant.context.delivery,
                    config_digest=config_digest,
                    snapshot=snapshot,
                )
    return list(targets.values())


def _fixture_repository_path(
    repository: Any,
    repo_root: Path,
) -> Path:
    if not isinstance(repository, FixtureRepositorySpec):
        return repo_root
    path = (repo_root / repository.path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError("fixture repository escapes the repository root") from exc
    actual = fixture_repository_digest(path)
    if actual != repository.sha256:
        raise ValueError(
            f"fixture repository digest changed: expected {repository.sha256}, got {actual}"
        )
    return path


def _experiment_with_request_overrides(
    experiment: ExperimentSpec,
    request: ExperimentRequest,
) -> ExperimentSpec:
    variant_ids = set(request.variants)
    missing = sorted(variant_ids - {variant.id for variant in experiment.variants})
    if missing:
        raise ValueError(f"unknown variant(s): {', '.join(missing)}")
    # A request narrows the plan, not the authored experiment. Removing variants here
    # invalidates unrelated workload contracts before workload selection runs.
    return experiment_with_overrides(
        experiment,
        model=request.model,
        builder_model=request.builder_model,
        judge_model=request.judge_model,
        # Request tags describe the execution context (for example campaign,
        # proposal, and stage). They must extend rather than replace the
        # experiment's authored semantic tags because saved analyses may use
        # those tags to select the exact registered treatment family.
        tags=list(dict.fromkeys([*experiment.tags, *request.tags])),
        harnesses=list(request.harnesses),
        n_tasks=request.n_tasks,
        n_attempts=request.n_attempts,
        n_concurrent=request.n_concurrent,
        trace_content=request.trace_content,
    )


def load_env(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key not in env:
            env[key] = value.strip().strip("'\"")
    return env


def as_json(value: Any) -> str:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, indent=2, sort_keys=True, default=_json_default)


def _merge_published_evaluations(
    existing: Any,
    published: tuple[PublishedEvaluation, ...],
) -> list[dict[str, Any]]:
    current = [dict(item) for item in (existing or []) if isinstance(item, dict)]
    replacements: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in published:
        value = asdict(item)
        replacements[_published_evaluation_key(value)] = value
    merged: list[dict[str, Any]] = []
    replaced: set[tuple[str, str, str]] = set()
    for item in current:
        key = _published_evaluation_key(item)
        if key not in replacements:
            merged.append(item)
            continue
        replacement = replacements[key]
        old_revision = int(item.get("revision") or 1)
        new_revision = int(replacement.get("revision") or 1)
        if old_revision < new_revision:
            merged.append({**item, "active": False})
            if key not in replaced:
                merged.append(replacement)
                replaced.add(key)
            continue
        if key not in replaced:
            merged.append(replacement)
            replaced.add(key)
    merged.extend(value for key, value in replacements.items() if key not in replaced)
    return merged


def _publication_manifest_update(
    manifest: dict[str, Any],
    evaluations: tuple[PublishedEvaluation, ...],
    failures: tuple[str, ...],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if evaluations:
        values["evaluation_runs"] = _merge_published_evaluations(
            manifest.get("evaluation_runs"), evaluations
        )
    if failures:
        values["evaluation_failures"] = list(
            dict.fromkeys(
                [
                    *(
                        str(item)
                        for item in manifest.get("evaluation_failures", [])
                        if item
                    ),
                    *(str(item) for item in failures if item),
                ]
            )
        )
    return values


def _published_evaluation_key(value: dict[str, Any]) -> tuple[str, str, str]:
    agent_predictions = int(value.get("agent_predictions") or 0)
    direct_predictions = int(value.get("direct_predictions") or 0)
    if agent_predictions and direct_predictions:
        kind = "mixed"
    elif direct_predictions:
        kind = "direct"
    elif agent_predictions:
        kind = "agent"
    else:
        kind = "unknown"
    identity = value.get("publication_id") or value.get("name") or ""
    return str(value.get("candidate_id") or ""), str(identity), kind


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


def _plan_coordinates(cells: tuple[PlannedCell, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            cell.workload_id,
            cell.task_id,
            cell.harness,
            cell.context_system_id,
            cell.context_delivery,
            cell.variant_id,
            cell.model_provider,
            cell.model,
            cell.trial_index,
            cell.comparison_example_id,
            cell.candidate_id,
            cell.execution_fingerprint,
            cell.execution_kind,
            cell.applicable,
            cell.skip_reason,
        )
        for cell in cells
    )


def _cancel_live_evaluation(
    live: LiveEvaluationCoordinator | None, message: str
) -> None:
    if live is None:
        return
    live.cancel_open_predictions(message)
    live.finalize(cancelled=True)


def _finalize_run(
    repo_root: Path,
    run_id: str,
    *,
    run_dir: Path,
    status: Literal["passed", "failed", "cancelled", "interrupted"],
    error: str | None,
    running: bool,
    values: dict[str, Any] | None = None,
) -> None:
    if running and status in {"failed", "cancelled", "interrupted"}:
        mark_unfinished_cells(run_dir, status, message=error or f"Run {status}")
    write_run_manifest(
        repo_root,
        run_id,
        {
            "status": status,
            "ended_at": _now(),
            "error": error,
            **(values or {}),
        },
    )


class _RunCancellation(Exception):
    pass


def _install_worker_sigterm_handler(
    cancellation_event: threading.Event,
) -> Callable[[], None]:
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    previous = signal.getsignal(signal.SIGTERM)

    def request_cancellation(signum: int, frame: Any) -> None:
        del signum, frame
        cancellation_event.set()

    signal.signal(signal.SIGTERM, request_cancellation)

    def restore() -> None:
        signal.signal(signal.SIGTERM, previous)

    return restore


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require_saved_evaluation_assets(
    repo_root: Path,
    experiment: ExperimentSpec,
    request: ExperimentRequest,
    workloads: list[WorkloadSpec],
) -> None:
    generated = (
        any(
            any(
                not scorer_reference(scorer).startswith("builtin:")
                for scorer in workload.scorers
            )
            for workload in workloads
        )
        or experiment.evaluation_generation is not None
    )
    if not generated:
        return
    if not (request.judge_model or experiment.judge_model):
        raise ValueError(
            "generated evaluation requires an explicit judge_model; run "
            f"`fugue plan {experiment.id}` to review and save the evaluation"
        )
    required: list[Path] = []
    for workload in workloads:
        if workload.runner != "harbor":
            continue
        manifest_path = _resolve(
            repo_root,
            request.manifest or workload.manifest or experiment.manifest,
        )
        required.append(manifest_path)
        required.extend(
            _resolve(repo_root, Path(scorer_reference(scorer)))
            for scorer in workload.scorers
            if not scorer_reference(scorer).startswith("builtin:")
        )
        if manifest_path.is_file():
            manifest = load_manifest(manifest_path)
            if manifest.dataset.materializer == (
                "fugue.bench.evaluations:GeneratedCapabilityMaterializer"
            ):
                for key in ("path", "rubric"):
                    if manifest.dataset.source.get(key):
                        required.append(
                            _resolve(
                                repo_root,
                                Path(str(manifest.dataset.source[key])),
                            )
                        )
    missing = list(dict.fromkeys(path for path in required if not path.is_file()))
    if not missing:
        return
    values = ", ".join(
        path.relative_to(repo_root).as_posix()
        if path.is_relative_to(repo_root)
        else path.as_posix()
        for path in missing
    )
    raise ValueError(
        f"evaluation draft is incomplete or unsaved ({values}); run "
        f"`fugue plan {experiment.id}` to generate, review, and save it"
    )


def _joined(values: tuple[str, ...]) -> str | None:
    return ",".join(values) if values else None


def _run_name(value: str | None, env: dict[str, str]) -> str:
    selected = value or env.get("FUGUE_RUN_NAME")
    if selected and selected.strip():
        return selected.strip()
    return datetime.now(UTC).strftime("fugue-%Y%m%dT%H%M%SZ")


def _run_tags(
    *,
    env: dict[str, str],
    tags: tuple[str, ...],
    run_name: str,
    manifest: BenchmarkManifest,
    manifest_path: Path,
) -> list[str]:
    configured = [
        part.strip() for part in env.get("FUGUE_TAGS", "").split(",") if part.strip()
    ]
    return _dedupe(
        [
            "fugue",
            f"run:{run_name}",
            f"dataset:{manifest.dataset.ref or manifest.dataset.path}",
            f"manifest:{manifest_path.stem}",
            *configured,
            *tags,
        ]
    )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _slug(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    return "-".join(part for part in out.split("-") if part) or "fugue"


def _env_id(value: str) -> str:
    return "".join(ch.upper() if ch.isalnum() else "_" for ch in value)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _selection_lock_digest(path: Path | None, repo_root: Path) -> str:
    if path is None:
        return ""
    resolved = _resolve(repo_root, path)
    if not resolved.is_file():
        raise ValueError(f"treatment selection lock does not exist: {resolved}")
    return read_treatment_selection_lock(resolved).lock_sha256


def _request_with_selection_lock(
    experiment: ExperimentSpec,
    request: ExperimentRequest,
    repo_root: Path,
) -> ExperimentRequest:
    preset = select_preset(experiment, request.preset)
    if not preset.selection_lock_required:
        return request
    if request.selection_lock is None:
        raise ValueError(f"preset {preset.id} requires --selection-lock")
    path = _resolve(repo_root, request.selection_lock)
    lock = read_treatment_selection_lock(path)
    selected = ("none", *lock.selected_variants)
    if request.variants and set(request.variants) != set(selected):
        raise ValueError(
            "requested variants disagree with the treatment selection lock: "
            + ", ".join(selected)
        )
    return replace(request, variants=selected)


def _run_job_paths(root: Path, run: ManagedRun) -> list[Path]:
    """Return only Harbor job roots owned by this immutable run."""
    configured = [Path(str(path)) for path in run.metadata.get("job_paths", [])]
    if not configured:
        configured = [
            Path(str(record["result_path"])).parent
            for record in latest_cell_records(run.run_dir / "cells.jsonl")
            if record.get("result_path")
        ]
    if not configured:
        # Schema-v1 runs created before exact job paths were recorded can only
        # fall back to the shared parent. New runs must never use this branch.
        configured = [Path(str(path)) for path in run.metadata.get("jobs_dirs", [])]
    return list(dict.fromkeys(_resolve(root, path) for path in configured))


def _agent_trace_refs(rows: list[dict[str, Any]]) -> tuple[AgentTraceRef, ...]:
    grouped: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        agent = str(row.get("weave_agent_name") or row.get("harness") or "agent")
        values = grouped.setdefault(
            agent,
            {"conversations": set(), "traces": set(), "roots": set()},
        )
        for key, target in (
            ("weave_conversation_ids", "conversations"),
            ("weave_trace_ids", "traces"),
            ("weave_root_span_ids", "roots"),
        ):
            raw = row.get(key) or []
            if isinstance(raw, str):
                raw = [raw]
            values[target].update(str(value) for value in raw if value)
        if row.get("weave_conversation_id"):
            values["conversations"].add(str(row["weave_conversation_id"]))
    return tuple(
        AgentTraceRef(
            agent_name=agent,
            conversation_ids=tuple(sorted(values["conversations"])),
            trace_ids=tuple(sorted(values["traces"])),
            root_span_ids=tuple(sorted(values["roots"])),
        )
        for agent, values in sorted(grouped.items())
    )


def _row_matches_cell(row: dict[str, Any], cell: CellSummary) -> bool:
    task_name = str(row.get("task_name") or "")
    return (
        str(row.get("harness") or "") == cell.harness
        and str(row.get("variant_id") or "baseline") == cell.variant_id
        and str(row.get("context_system_id") or "none") == cell.context_system_id
        and (task_name == cell.task_id or task_name.endswith(f"/{cell.task_id}"))
    )


def _merge_report_rows(
    rows: list[dict[str, Any]], reports_root: Path
) -> list[dict[str, Any]]:
    merged = {_result_identity(row): row for row in rows}
    if reports_root.is_dir():
        for path in sorted(reports_root.glob("*.jsonl")):
            for line in path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("record_type"):
                    merged[_result_identity(row)] = row
    return list(merged.values())


def _result_identity(row: dict[str, Any]) -> tuple[str, str]:
    record_type = str(row.get("record_type") or "unknown")
    identity = (
        row.get("run_key")
        or row.get("cell_id")
        or row.get("query_id")
        or row.get("probe_id")
        or json.dumps(row, sort_keys=True, default=str)
    )
    return record_type, str(identity)


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
