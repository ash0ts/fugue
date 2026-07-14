from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fugue.bench.context import (
    CONTEXT_MANIFEST,
    DEFAULT_CACHE_ROOT,
    list_context_systems,
)
from fugue.bench.execution import latest_cell_records, new_run_id
from fugue.bench.export import export_rows
from fugue.bench.library import ExperimentSpec, get_experiment, list_experiments
from fugue.bench.supervisor import ManagedRun, RunSupervisor
from fugue.bridge import bridge_status
from fugue.model_plane import resolve_model_route, select_model, trace_project_slug


@dataclass(frozen=True)
class ExperimentRequest:
    experiment_id: str = "pilot"
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


@dataclass(frozen=True)
class CellSummary:
    cell_id: str
    status: str
    harness: str
    variant_id: str
    context_system_id: str
    workload_id: str
    task_id: str
    wall_time_sec: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class AgentTraceRef:
    agent_name: str
    conversation_ids: tuple[str, ...] = ()
    trace_ids: tuple[str, ...] = ()
    root_span_ids: tuple[str, ...] = ()


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
    pending: int
    not_applicable: int
    log_path: Path


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
    rows: tuple[dict[str, Any], ...] = ()
    agent_traces: tuple[AgentTraceRef, ...] = ()


@dataclass(frozen=True)
class DeepLinks:
    project: str
    weave: str
    agents: str
    trace: str | None = None


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
    catalog_records: int = 0
    catalog_refreshed_at: str | None = None
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

    def status(self, request: ExperimentRequest | None = None) -> OperatorStatus:
        env = self.env
        experiment = self.experiment(request.experiment_id if request else "pilot")
        selected_model = select_model(
            request.model if request else None,
            env=env,
            experiment_model=experiment.model,
        )
        route = resolve_model_route(selected_model, env)
        builder_model = (
            (request.builder_model if request else None)
            or experiment.builder_model
            or env.get("FUGUE_BUILDER_MODEL")
            or selected_model
        )
        judge_model = (
            (request.judge_model if request else None)
            or experiment.judge_model
            or env.get("FUGUE_JUDGE_MODEL")
        )
        routes = [self._role_status("target", selected_model, env)]
        routes.append(self._role_status("builder", builder_model, env))
        if judge_model:
            routes.append(self._role_status("judge", judge_model, env))
        from fugue.assistant import select_assistant_model
        from fugue.bench.catalog import ExperimentCatalog

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
                    "analyst", experiment_model=selected_model, env=env
                ),
                env,
            )
        )
        trace_project = trace_project_slug(env)
        bridge = bridge_status()
        trace_content = (
            (request.trace_content if request else None) or experiment.trace_content
        )
        key_names = ("WANDB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        selected_systems = tuple(
            dict.fromkeys(
                (request.systems if request and request.systems else ())
                or tuple(
                    variant.context.system_id
                    for variant in experiment.variants
                    if variant.enabled
                )
            )
        )
        cache_root = self.repo_root / DEFAULT_CACHE_ROOT
        catalog_status = ExperimentCatalog(self.repo_root, env).status()
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
            catalog_records=catalog_status.records,
            catalog_refreshed_at=catalog_status.refreshed_at,
            keys={key: bool(env.get(key, "").strip()) for key in key_names},
        )

    def experiment(self, experiment_id: str) -> ExperimentSpec:
        return get_experiment(experiment_id, self.repo_root)

    def experiment_items(self) -> list[tuple[str, str]]:
        return [(item.title, item.id) for item in list_experiments(self.repo_root)]

    def preview(self, request: ExperimentRequest) -> PreviewSummary:
        return self._preview_namespace(self._namespace(request))

    def preview_experiment(self, experiment: ExperimentSpec) -> PreviewSummary:
        namespace = self._namespace(
            ExperimentRequest(
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
        )
        namespace.experiment_spec = experiment
        return self._preview_namespace(namespace)

    def _preview_namespace(self, namespace: Any) -> PreviewSummary:
        from fugue.bench.cli import _rendered_jobs_from_args

        jobs = _rendered_jobs_from_args(
            namespace,
            run_id="preview",
            write_configs=False,
        )
        estimated_trials = 0
        for job in jobs:
            if not job.applicable:
                continue
            task_count = int((job.config.get("fugue") or {}).get("task_count") or 1)
            attempts = int(
                job.config.get("n_attempts")
                or (job.config.get("fugue") or {}).get("n_attempts")
                or 1
            )
            estimated_trials += task_count * attempts
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
            commands=tuple(" ".join(job.command) for job in jobs),
        )

    def launch_experiment(
        self,
        experiment: ExperimentSpec,
        *,
        attached: bool,
    ) -> RunSummary:
        from fugue.bench.library import experiment_to_yaml

        run_id = new_run_id()
        run_dir = self.repo_root / ".fugue" / "runtime" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot = run_dir / "experiment.yaml"
        temp = snapshot.with_suffix(".tmp")
        temp.write_text(experiment_to_yaml(experiment))
        os.replace(temp, snapshot)
        run_name = experiment.run_name or experiment.id
        command = [
            sys.executable,
            "-m",
            "fugue.bench.cli",
            "run",
            "--experiment-file",
            snapshot.as_posix(),
            "--repo-root",
            self.repo_root.as_posix(),
            "--env-file",
            self.env_file.as_posix(),
            "--run-id",
            run_id,
        ]
        self.supervisor.start_detached(
            run_id=run_id,
            command=command,
            env=self.env,
            run_name=run_name,
            experiment_id=experiment.id,
        )
        return self.run_summary(run_id)

    async def compose_experiment(
        self,
        request: str,
        *,
        base_experiment: str = "pilot",
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

    async def analyze_experiments(
        self,
        question: str,
        *,
        filters: dict[str, str] | None = None,
        model: str | None = None,
        source: str | None = None,
    ) -> Any:
        from fugue.bench.ai import ExperimentAnalyst

        return await ExperimentAnalyst(self).analyze(
            question,
            filters=filters,
            model=model,
            source=source,
        )

    def launch(self, request: ExperimentRequest, *, attached: bool) -> RunSummary:
        run_id = new_run_id()
        experiment = self.experiment(request.experiment_id)
        run_name = request.run_name or experiment.run_name or experiment.id
        command = [
            sys.executable,
            "-m",
            "fugue.bench.cli",
            *self.request_arguments(request),
            "--run-id",
            run_id,
        ]
        run = self.supervisor.start_detached(
            run_id=run_id,
            command=command,
            env=self.env,
            run_name=run_name,
            experiment_id=request.experiment_id,
        )
        if attached:
            # Attached means the operator follows the durable run; the child
            # still owns its own process group so closing the TUI is safe.
            return self.run_summary(run.run_id)
        return self.run_summary(run.run_id)

    def request_arguments(self, request: ExperimentRequest) -> list[str]:
        args = ["run", "--experiment", request.experiment_id]
        for flag, value in (
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

    def runs(self) -> list[RunSummary]:
        return [self._summarize_run(run) for run in self.supervisor.list()]

    def run_summary(self, run_id: str) -> RunSummary:
        return self._summarize_run(self.supervisor.get(run_id))

    def run_trace_refs(
        self, run_id: str, *, cell_id: str | None = None
    ) -> tuple[AgentTraceRef, ...]:
        run = self.supervisor.get(run_id)
        sources = [
            self.repo_root / Path(str(path))
            for path in run.metadata.get("jobs_dirs", [])
        ]
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

    def results(self, paths: list[Path] | None = None) -> ResultSummary:
        sources = paths or [
            self.repo_root / "jobs",
            self.repo_root / ".fugue" / "runtime",
        ]
        rows = export_rows([path for path in sources if path.exists()])
        if paths is None:
            rows = _merge_report_rows(rows, self.repo_root / "reports")
        trials = [row for row in rows if row.get("record_type") == "trial"]
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
            tool_calls=sum(int(row.get("weave_tool_call_count") or 0) for row in trials),
            turns=sum(int(row.get("weave_turn_count") or 0) for row in trials),
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
        run = self.supervisor.get(run_id)
        project = str(run.metadata.get("trace_project") or "").strip() or None
        return self.deep_links(project=project)

    def _namespace(self, request: ExperimentRequest) -> Any:
        from argparse import Namespace

        return Namespace(
            experiment=request.experiment_id,
            manifest=None,
            harnesses=_joined(request.harnesses),
            variants=_joined(request.variants),
            preset=request.preset,
            workloads=_joined(request.workloads),
            systems=_joined(request.systems),
            model=request.model,
            judge_model=request.judge_model,
            builder_model=request.builder_model,
            n_attempts=request.n_attempts,
            n_concurrent=request.n_concurrent,
            n_tasks=request.n_tasks,
            env_file=self.env_file,
            jobs_dir=request.jobs_dir,
            run_name=request.run_name,
            tags=_joined(request.tags),
            repo_root=self.repo_root,
            trace_content=request.trace_content,
            experiment_spec=None,
        )

    @staticmethod
    def _role_status(
        role: str, model: str, env: dict[str, str]
    ) -> ModelRoleStatus:
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
            )
            for item in records
        )
        return RunSummary(
            run_id=run.run_id,
            run_name=run.run_name,
            experiment_id=run.experiment_id,
            status=run.status,
            created_at=run.created_at,
            cells=cells,
            passed=sum(cell.status == "passed" for cell in cells),
            failed=sum(cell.status in {"failed", "cancelled", "interrupted"} for cell in cells),
            pending=sum(cell.status in {"pending", "running"} for cell in cells),
            not_applicable=sum(cell.status == "not_applicable" for cell in cells),
            log_path=run.log_path,
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
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _joined(values: tuple[str, ...]) -> str | None:
    return ",".join(values) if values else None


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
