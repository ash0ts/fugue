from __future__ import annotations

import difflib
import hashlib
import json
import re
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from fugue.assistant import (
    AssistantAgent,
    AssistantMessage,
    AssistantModelClient,
    AssistantRunResult,
    AssistantTool,
    select_assistant_model,
)
from fugue.bench.catalog import FILTER_FIELDS, ArtifactExcerpt, ExperimentCatalog
from fugue.bench.context import get_context_system, list_context_systems
from fugue.bench.export import fetch_weave_summaries
from fugue.bench.integrations import list_integrations, load_integration
from fugue.bench.library import (
    ExperimentSpec,
    experiment_from_data,
    experiment_to_yaml,
    get_experiment,
    get_prompt,
    get_skill,
    list_experiments,
    list_prompts,
    list_skills,
    validate_id,
)
from fugue.bench.manifest import load_manifest
from fugue.bench.sources import list_skill_source_ids, load_skill_source
from fugue.model_plane import resolve_model_route, select_model, trace_project_slug

if TYPE_CHECKING:
    from fugue.bench.operator import OperatorService, PreviewSummary

ANALYSES_DIR = Path("configs/fugue/analyses")
ANALYSIS_REPORTS_DIR = Path("reports/analyses")
AnalysisSource = Literal["local", "hybrid"]
ClientFactory = Callable[[str, Mapping[str, str]], AssistantModelClient]
ANALYSIS_GROUP_FIELDS = FILTER_FIELDS | {
    "skill_ids",
    "integration_ids",
    "manifest",
    "run_name",
}


@dataclass(frozen=True)
class AssetDraft:
    kind: Literal["prompt", "skill"]
    id: str
    title: str
    body: str


@dataclass(frozen=True)
class ExperimentDraft:
    experiment: ExperimentSpec
    assets: tuple[AssetDraft, ...]
    rationale: str
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    diff: str
    preview: PreviewSummary
    model: str
    provider: str
    session_id: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class AnalysisSpec:
    id: str
    title: str
    question: str
    filters: dict[str, str] = field(default_factory=dict)
    group_by: tuple[str, ...] = (
        "experiment_id",
        "harness",
        "variant_id",
        "context_system_id",
    )
    metrics: tuple[str, ...] = (
        "pass_rate",
        "reward",
        "wall_time_sec",
        "cost_usd",
        "tokens",
        "failures",
        "tool_calls",
    )
    source: AnalysisSource = "hybrid"
    include_artifacts: bool = True
    model: str | None = None
    trace_content: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisScope:
    experiments: tuple[str, ...]
    runs: tuple[str, ...]
    rows: int
    tasks: tuple[str, ...]
    models: tuple[str, ...]
    variants: tuple[str, ...]
    sources: tuple[str, ...]
    missing_metrics: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisSnapshot:
    id: str
    digest: str
    created_at: str
    catalog_revision: str | None
    row_ids: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EvidenceRef:
    id: str
    kind: str
    label: str
    row_ids: tuple[str, ...] = ()
    local_path: str | None = None
    weave_call_id: str | None = None
    conversation_id: str | None = None
    sha256: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    spec: AnalysisSpec
    scope: AnalysisScope
    snapshot: AnalysisSnapshot
    evidence: tuple[EvidenceRef, ...]
    aggregates: tuple[dict[str, Any], ...]
    report: str
    report_dir: Path
    model: str
    provider: str
    session_id: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class AnalysisPreview:
    spec: AnalysisSpec
    scope: AnalysisScope
    snapshot: AnalysisSnapshot
    evidence: tuple[EvidenceRef, ...]
    aggregates: tuple[dict[str, Any], ...]


class ExperimentComposer:
    def __init__(
        self,
        operator: OperatorService,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.operator = operator
        self.repo_root = operator.repo_root
        self.client_factory = client_factory or _client_factory

    async def compose(
        self,
        request: str,
        *,
        base_experiment: str = "pilot",
        model: str | None = None,
        trace_content: str | None = None,
    ) -> ExperimentDraft:
        base = get_experiment(base_experiment, self.repo_root)
        env = self.operator.env
        selected_model = select_assistant_model(
            "composer",
            cli_model=model,
            experiment_model=base.model,
            env=env,
        )
        route = resolve_model_route(selected_model, env)
        if not env.get(route.api_key_env, "").strip():
            raise RuntimeError(f"{route.api_key_env} is required for the composer")
        client = self.client_factory(selected_model, env)
        tools = self._tools(base)
        agent = AssistantAgent(
            client,
            role="composer",
            tools=tools,
            env=env,
            trace_content=trace_content or base.trace_content,
            max_rounds=8,
            attributes={
                "fugue.ai.base_experiment": base.id,
                "fugue.ai.action": "compose",
            },
        )
        messages = [
            AssistantMessage("system", _composer_instructions()),
            AssistantMessage(
                "user",
                json.dumps(
                    {
                        "request": request,
                        "base_experiment": base.to_dict(),
                        "available": self._catalog_summary(base),
                    },
                    sort_keys=True,
                    default=str,
                ),
            ),
        ]
        base_messages = list(messages)
        validation_error: Exception | None = None
        for attempt in range(3):
            result = await agent.run(messages)
            try:
                draft = self._validate_draft(result, base)
            except Exception as exc:
                validation_error = exc
                if attempt == 2:
                    break
                messages = [
                    *base_messages,
                    AssistantMessage(
                        "user",
                        "The proposed draft failed deterministic validation. "
                        f"Correct it and call submit_experiment again. Error: {exc}",
                    ),
                ]
                continue
            return draft
        assert validation_error is not None
        raise ValueError(f"composer draft remained invalid: {validation_error}")

    def save(
        self,
        draft: ExperimentDraft,
        *,
        experiment_id: str,
        replace_assets: bool = False,
    ) -> ExperimentSpec:
        return self.operator.save_working_experiment(
            draft.experiment,
            self.operator.request_for_experiment(draft.experiment),
            experiment_id=experiment_id,
            assets=draft.assets,
            replace_assets=replace_assets,
        )

    def _tools(self, base: ExperimentSpec) -> tuple[AssistantTool, ...]:
        return (
            AssistantTool(
                "list_fugue_assets",
                "List the repository-backed experiment assets currently available.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                lambda _: self._catalog_summary(base),
            ),
            AssistantTool(
                "show_experiment",
                "Load one complete saved Fugue experiment by id.",
                {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
                lambda value: get_experiment(str(value["id"]), self.repo_root).to_dict(),
            ),
            AssistantTool(
                "submit_experiment",
                "Submit a complete Fugue experiment draft for deterministic validation.",
                _experiment_submission_schema(),
                terminal=True,
            ),
        )

    def _catalog_summary(self, base: ExperimentSpec) -> dict[str, Any]:
        experiments: list[dict[str, Any]] = []
        for item in list_experiments(self.repo_root):
            experiment = get_experiment(item.id, self.repo_root)
            experiments.append(
                {
                    "id": experiment.id,
                    "title": experiment.title,
                    "model": experiment.model,
                    "harnesses": experiment.harnesses,
                    "workloads": [
                        {
                            "id": workload.id,
                            "runner": workload.runner,
                            "manifest": (
                                workload.manifest.as_posix()
                                if workload.manifest
                                else None
                            ),
                            "systems": workload.systems,
                        }
                        for workload in experiment.workloads
                    ],
                    "presets": [
                        {
                            "id": preset.id,
                            "workloads": preset.workloads,
                            "systems": preset.systems,
                            "harnesses": preset.harnesses,
                        }
                        for preset in experiment.presets
                    ],
                }
            )

        manifests: list[dict[str, Any]] = []
        for path in sorted((self.repo_root / "datasets").rglob("*.yaml")):
            relative = path.relative_to(self.repo_root).as_posix()
            try:
                manifest = load_manifest(path)
            except (OSError, TypeError, ValueError):
                manifests.append({"path": relative, "valid": False})
                continue
            manifests.append(
                {
                    "path": relative,
                    "valid": True,
                    "dataset": manifest.dataset.harbor_ref,
                    "tasks": [task.id for task in manifest.tasks],
                    "harnesses": [harness.name for harness in manifest.harnesses],
                }
            )

        env = self.operator.env
        target = select_model(env=env, experiment_model=base.model)
        role_models = {
            "target": target,
            "composer": select_assistant_model(
                "composer", experiment_model=target, env=env
            ),
            "analyst": select_assistant_model(
                "analyst", experiment_model=target, env=env
            ),
        }
        routes: dict[str, dict[str, Any]] = {}
        for role, model in role_models.items():
            route = resolve_model_route(model, env)
            routes[role] = {
                "model": route.display_model,
                "provider": route.provider,
                "key_env": route.api_key_env,
                "key_present": bool(env.get(route.api_key_env, "").strip()),
            }
        return {
            "experiments": experiments,
            "prompts": [item.id for item in list_prompts(self.repo_root)],
            "skills": sorted(
                {
                    *[item.id for item in list_skills(self.repo_root)],
                    *list_skill_source_ids(self.repo_root),
                }
            ),
            "integrations": [
                item.id for item in list_integrations(self.repo_root)
            ],
            "context_systems": [item.id for item in list_context_systems(self.repo_root)],
            "manifests": manifests,
            "harnesses": ["hermes", "openclaw", "claude-code", "codex"],
            "model_prefixes": ["wandb/", "openai/", "anthropic/"],
            "routes": routes,
            "trace_key_present": bool(env.get("WANDB_API_KEY", "").strip()),
        }

    def _validate_draft(
        self,
        result: AssistantRunResult,
        base: ExperimentSpec,
    ) -> ExperimentDraft:
        raw = result.payload
        experiment_raw = raw.get("experiment")
        if not isinstance(experiment_raw, dict):
            raise ValueError("submit_experiment requires an experiment object")
        assets = tuple(_asset_draft(item) for item in raw.get("assets") or [])
        experiment = experiment_from_data(experiment_raw)
        _validate_experiment_references(experiment, assets, self.repo_root, self.operator.env)
        preview = self.operator.preview_experiment(experiment)
        diff = "\n".join(
            difflib.unified_diff(
                experiment_to_yaml(base).splitlines(),
                experiment_to_yaml(experiment).splitlines(),
                fromfile=f"{base.id}.yaml",
                tofile=f"{experiment.id}.yaml",
                lineterm="",
            )
        )
        return ExperimentDraft(
            experiment=experiment,
            assets=assets,
            rationale=str(raw.get("rationale") or ""),
            assumptions=tuple(str(item) for item in raw.get("assumptions") or []),
            warnings=tuple(str(item) for item in raw.get("warnings") or []),
            diff=diff,
            preview=preview,
            model=result.model,
            provider=result.provider,
            session_id=result.session_id,
            input_tokens=result.usage.input_tokens or 0,
            output_tokens=result.usage.output_tokens or 0,
        )


class ExperimentAnalyst:
    def __init__(
        self,
        operator: OperatorService,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.operator = operator
        self.repo_root = operator.repo_root
        self.client_factory = client_factory or _client_factory
        self.catalog = ExperimentCatalog(self.repo_root)

    async def analyze(
        self,
        question: str | None = None,
        *,
        spec: AnalysisSpec | None = None,
        filters: Mapping[str, str] | None = None,
        model: str | None = None,
        source: AnalysisSource | None = None,
    ) -> AnalysisResult:
        if spec is None:
            if not question or not question.strip():
                raise ValueError("analysis question is required")
            spec = await self.plan(
                question,
                filters=filters,
                model=model,
                source=source,
            )
        elif filters:
            spec = AnalysisSpec(**{**spec.to_dict(), "filters": {**spec.filters, **filters}})
        preview = self.prepare(spec)
        return await self.execute(preview, model=model)

    async def plan(
        self,
        question: str,
        *,
        filters: Mapping[str, str] | None = None,
        model: str | None = None,
        source: AnalysisSource | None = None,
    ) -> AnalysisSpec:
        base_experiment = None
        if filters and filters.get("experiment_id"):
            try:
                base_experiment = get_experiment(filters["experiment_id"], self.repo_root)
            except FileNotFoundError:
                pass
        env = self.operator.env
        selected_model = select_assistant_model(
            "analyst",
            cli_model=model,
            experiment_model=base_experiment.model if base_experiment else None,
            env=env,
        )
        client = self.client_factory(selected_model, env)
        self.catalog.refresh()
        catalog = self.catalog.experiment_catalog()
        facets = self.catalog.facets()
        agent = AssistantAgent(
            client,
            role="analyst",
            tools=(
                AssistantTool(
                    "list_experiment_catalog",
                    "Return saved experiment metadata and deterministic intervention buckets.",
                    {"type": "object", "properties": {}, "additionalProperties": False},
                    lambda _: {"experiments": catalog, "facets": facets},
                ),
                AssistantTool(
                    "submit_analysis_plan",
                    "Submit a grounded analysis scope and deterministic metric plan.",
                    _analysis_plan_schema(),
                    terminal=True,
                ),
            ),
            env=env,
            trace_content=base_experiment.trace_content if base_experiment else "full",
            attributes={
                "fugue.ai.action": "plan_analysis",
                "fugue.ai.experiment_ids": sorted(
                    {
                        str(item.get("experiment_id"))
                        for item in catalog
                        if item.get("experiment_id")
                    }
                ),
            },
        )
        base_messages = [
            AssistantMessage("system", _analyst_planner_instructions()),
            AssistantMessage(
                "user",
                json.dumps(
                    {
                        "question": question,
                        "required_filters": dict(filters or {}),
                        "requested_source": source,
                        "experiments": catalog,
                        "facets": facets,
                    },
                    sort_keys=True,
                    default=str,
                ),
            ),
        ]
        error: Exception | None = None
        for attempt in range(3):
            messages = list(base_messages)
            if error is not None:
                messages.append(
                    AssistantMessage(
                        "user",
                        f"The analysis plan failed deterministic validation. Correct it. Error: {error}",
                    )
                )
            run = await agent.run(messages)
            payload = run.payload
            try:
                resolved_filters = {
                    str(key): str(value)
                    for key, value in (payload.get("filters") or {}).items()
                }
                resolved_filters.update(
                    {str(key): str(value) for key, value in (filters or {}).items()}
                )
                return _analysis_spec(
                    {
                        **payload,
                        "question": question,
                        "filters": resolved_filters,
                        "source": source or payload.get("source") or "hybrid",
                        "model": selected_model,
                    }
                )
            except Exception as exc:
                error = exc
                if attempt == 2:
                    break
        assert error is not None
        raise ValueError(f"analysis plan remained invalid: {error}")

    def prepare(self, spec: AnalysisSpec) -> AnalysisPreview:
        status = self.catalog.refresh()
        filters = {**spec.filters, "record_type": "trial", "source": "local"}
        rows = self.catalog.records(filters=filters)
        if not rows:
            raise ValueError("analysis scope resolved to no experiment records")
        snapshot = _snapshot(rows, status.revision)
        scope = _scope(rows, spec.metrics, [])
        aggregates, evidence = _aggregate(rows, spec)
        return AnalysisPreview(
            spec=spec,
            scope=scope,
            snapshot=snapshot,
            evidence=tuple(evidence),
            aggregates=tuple(aggregates),
        )

    async def execute(
        self,
        preview: AnalysisPreview,
        *,
        model: str | None = None,
    ) -> AnalysisResult:
        spec = preview.spec
        rows = [dict(row) for row in preview.snapshot.rows]
        warnings: list[str] = []
        if spec.source == "hybrid":
            try:
                run_keys = [str(row["run_key"]) for row in rows if row.get("run_key")]
                conversations = {
                    str(row["run_key"]): [
                        str(value)
                        for value in (
                            row.get("weave_conversation_ids")
                            or row.get("native_session_ids")
                            or []
                        )
                        if value
                    ]
                    for row in rows
                    if row.get("run_key")
                }
                remote = fetch_weave_summaries(
                    run_keys,
                    conversation_ids_by_run=conversations,
                    project=trace_project_slug(self.operator.env),
                    env=self.operator.env,
                )
                rows = [
                    _merge_weave_record(row, remote.get(str(row.get("run_key"))))
                    for row in rows
                ]
            except Exception as exc:
                warnings.append(f"Weave enrichment unavailable; using local data: {exc}")
        snapshot = _snapshot(rows, preview.snapshot.catalog_revision)
        scope = _scope(rows, spec.metrics, warnings)
        aggregates, evidence = _aggregate(rows, spec)
        selected_model = select_assistant_model(
            "analyst",
            cli_model=model,
            saved_model=spec.model,
            experiment_model=_scope_experiment_model(rows, self.repo_root),
            env=self.operator.env,
        )
        client = self.client_factory(selected_model, self.operator.env)
        evidence_values = list(evidence)

        def read_artifact(arguments: dict[str, Any]) -> dict[str, Any]:
            if not spec.include_artifacts:
                raise ValueError("artifact inspection is disabled for this analysis")
            excerpt = self.catalog.read_artifact(str(arguments["path"]))
            ref = _artifact_evidence(excerpt, len(evidence_values) + 1)
            evidence_values.append(ref)
            return {"evidence": asdict(ref), "excerpt": excerpt.text}

        def inspect_failures(arguments: dict[str, Any]) -> dict[str, Any]:
            limit = min(max(int(arguments.get("limit") or 10), 1), 20)
            failures = [
                row
                for row in rows
                if row.get("pass") is False or row.get("exception_class")
            ][:limit]
            ref = EvidenceRef(
                id=f"E{len(evidence_values) + 1:03d}",
                kind="failures",
                label=f"{len(failures)} scoped failure records",
                row_ids=tuple(str(row.get("row_id")) for row in failures),
                data={"count": len(failures)},
            )
            evidence_values.append(ref)
            safe_rows = [
                {
                    key: row.get(key)
                    for key in (
                        "row_id",
                        "run_id",
                        "run_key",
                        "task_name",
                        "harness",
                        "variant_id",
                        "context_system_id",
                        "exception_class",
                        "exception_message",
                        "trial_dir",
                        "weave_conversation_ids",
                    )
                }
                for row in failures
            ]
            return {"evidence": asdict(ref), "failures": safe_rows}

        def inspect_weave_conversation(arguments: dict[str, Any]) -> dict[str, Any]:
            run_key = str(arguments["run_key"])
            row = next((item for item in rows if item.get("run_key") == run_key), None)
            if row is None:
                raise ValueError(f"run key is outside the analysis snapshot: {run_key}")
            conversation_ids = row.get("weave_conversation_ids") or []
            if isinstance(conversation_ids, str):
                conversation_ids = [conversation_ids]
            conversation_id = (
                next((str(item) for item in conversation_ids if item), None)
                or row.get("weave_conversation_id")
                or row.get("conversation_id")
            )
            call_id = row.get("weave_call_id") or row.get("call_id")
            if not call_id and not conversation_id:
                raise ValueError(
                    "the selected trial has no cataloged Weave conversation metadata"
                )
            ref = EvidenceRef(
                id=f"E{len(evidence_values) + 1:03d}",
                kind="weave_conversation",
                label=f"Weave conversation metadata for {run_key}",
                row_ids=(str(row.get("row_id")),),
                weave_call_id=str(call_id) if call_id else None,
                conversation_id=str(conversation_id) if conversation_id else None,
                data={
                    "trace_id": row.get("weave_trace_id") or row.get("trace_id"),
                    "summary": row.get("weave_summary") or {},
                },
            )
            evidence_values.append(ref)
            return {"evidence": asdict(ref)}

        agent = AssistantAgent(
            client,
            role="analyst",
            tools=(
                AssistantTool(
                    "get_analysis_scope",
                    "Return the immutable scope, aggregates, and available evidence ids.",
                    {"type": "object", "properties": {}, "additionalProperties": False},
                    lambda _: {
                        "scope": asdict(scope),
                        "aggregates": aggregates,
                        "evidence": [asdict(item) for item in evidence_values],
                    },
                ),
                AssistantTool(
                    "read_result_artifact",
                    "Read one bounded local result artifact and register it as evidence.",
                    {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                    read_artifact,
                ),
                AssistantTool(
                    "inspect_failures",
                    "Inspect up to twenty failed trial records and register them as evidence.",
                    {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "minimum": 1, "maximum": 20}
                        },
                        "additionalProperties": False,
                    },
                    inspect_failures,
                ),
                AssistantTool(
                    "inspect_weave_conversation",
                    "Inspect bounded call, trace, conversation, and summary metadata for one "
                    "run key in the immutable analysis snapshot.",
                    {
                        "type": "object",
                        "properties": {"run_key": {"type": "string"}},
                        "required": ["run_key"],
                        "additionalProperties": False,
                    },
                    inspect_weave_conversation,
                ),
                AssistantTool(
                    "submit_analysis_report",
                    "Submit evidence-backed claims and a concise conclusion.",
                    _analysis_report_schema(),
                    terminal=True,
                ),
            ),
            env=self.operator.env,
            trace_content=spec.trace_content,
            attributes={
                "fugue.ai.action": "analyze",
                "fugue.ai.analysis_id": spec.id,
                "fugue.ai.snapshot_digest": snapshot.digest,
                "fugue.ai.experiment_ids": list(scope.experiments),
                "fugue.ai.run_ids": list(scope.runs),
                "fugue.ai.source": spec.source,
            },
        )
        report_error: Exception | None = None
        run: AssistantRunResult | None = None
        report = ""
        for attempt in range(3):
            messages = [
                AssistantMessage("system", _analyst_report_instructions()),
                AssistantMessage(
                    "user",
                    json.dumps(
                        {
                            "question": spec.question,
                            "scope": asdict(scope),
                            "aggregates": aggregates,
                            "evidence": [asdict(item) for item in evidence_values],
                        },
                        sort_keys=True,
                        default=str,
                    ),
                ),
            ]
            if report_error is not None:
                messages.append(
                    AssistantMessage(
                        "user",
                        "The report failed evidence validation. Correct it using only "
                        f"the supplied evidence ids. Error: {report_error}",
                    )
                )
            run = await agent.run(messages)
            try:
                report = _render_report(spec, run.payload, evidence_values, scope)
                break
            except Exception as exc:
                report_error = exc
                if attempt == 2:
                    raise ValueError(f"analysis report remained invalid: {exc}") from exc
        assert run is not None
        report_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        report_dir = self.repo_root / ANALYSIS_REPORTS_DIR / spec.id / report_id
        result = AnalysisResult(
            spec=spec,
            scope=scope,
            snapshot=snapshot,
            evidence=tuple(evidence_values),
            aggregates=tuple(aggregates),
            report=report,
            report_dir=report_dir,
            model=run.model,
            provider=run.provider,
            session_id=run.session_id,
            input_tokens=run.usage.input_tokens or 0,
            output_tokens=run.usage.output_tokens or 0,
        )
        _write_analysis(result)
        return result


def list_analyses(repo_root: Path | None = None) -> list[dict[str, str]]:
    root = (repo_root or Path.cwd()) / ANALYSES_DIR
    if not root.is_dir():
        return []
    values: list[dict[str, str]] = []
    for path in sorted(root.glob("*.yaml")):
        spec = get_analysis(path.stem, repo_root)
        values.append({"id": spec.id, "title": spec.title, "path": path.as_posix()})
    return values


def get_analysis(item_id: str, repo_root: Path | None = None) -> AnalysisSpec:
    item_id = validate_id(item_id, kind="analysis id")
    path = (repo_root or Path.cwd()) / ANALYSES_DIR / f"{item_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"analysis not found: {item_id}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("analysis YAML must be a mapping")
    spec = _analysis_spec(raw)
    if spec.id != item_id:
        raise ValueError(f"analysis file {path.name} declares mismatched id {spec.id}")
    return spec


def save_analysis(spec: AnalysisSpec, repo_root: Path | None = None) -> Path:
    root = (repo_root or Path.cwd()) / ANALYSES_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{validate_id(spec.id, kind='analysis id')}.yaml"
    path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False))
    return path


def _analysis_spec(raw: Mapping[str, Any]) -> AnalysisSpec:
    item_id = validate_id(str(raw.get("id") or _slug(str(raw.get("title") or raw.get("question") or "analysis"))), kind="analysis id")
    source = str(raw.get("source") or "hybrid")
    if source not in {"local", "hybrid"}:
        raise ValueError("analysis source must be local or hybrid")
    trace_content = str(raw.get("trace_content") or "full")
    if trace_content not in {"full", "metadata"}:
        raise ValueError("analysis trace_content must be full or metadata")
    filters = raw.get("filters") or {}
    if not isinstance(filters, dict):
        raise ValueError("analysis filters must be a mapping")
    unknown_filters = sorted(set(filters) - FILTER_FIELDS)
    if unknown_filters:
        raise ValueError(f"unsupported analysis filter(s): {', '.join(unknown_filters)}")
    group_by = tuple(
        str(item)
        for item in raw.get("group_by")
        or AnalysisSpec.__dataclass_fields__["group_by"].default
    )
    unknown_groups = sorted(set(group_by) - ANALYSIS_GROUP_FIELDS)
    if unknown_groups:
        raise ValueError(
            f"unsupported analysis grouping field(s): {', '.join(unknown_groups)}"
        )
    return AnalysisSpec(
        id=item_id,
        title=str(raw.get("title") or item_id),
        question=str(raw.get("question") or "").strip(),
        filters={str(key): str(value) for key, value in filters.items()},
        group_by=group_by,
        metrics=tuple(str(item) for item in raw.get("metrics") or AnalysisSpec.__dataclass_fields__["metrics"].default),
        source=source,  # type: ignore[arg-type]
        include_artifacts=bool(raw.get("include_artifacts", True)),
        model=str(raw["model"]) if raw.get("model") else None,
        trace_content=trace_content,
    )


def _aggregate(
    rows: Sequence[dict[str, Any]], spec: AnalysisSpec
) -> tuple[list[dict[str, Any]], list[EvidenceRef]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key) or "unknown") for key in spec.group_by)].append(row)
    aggregates: list[dict[str, Any]] = []
    evidence: list[EvidenceRef] = []
    for index, (key, values) in enumerate(sorted(groups.items()), start=1):
        scored = [row for row in values if row.get("pass") is not None]
        rewards = _numbers(values, "reward")
        latencies = _numbers(values, "wall_time_sec")
        costs = _numbers(values, "cost_usd")
        payload = {
            **dict(zip(spec.group_by, key, strict=True)),
            "trials": len(values),
            "scored_trials": len(scored),
            "pass_rate": (
                sum(row.get("pass") is True for row in scored) / len(scored)
                if scored
                else None
            ),
            "average_reward": _average(rewards),
            "average_wall_time_sec": _average(latencies),
            "total_cost_usd": sum(costs) if costs else None,
            "total_tokens": sum(
                int(row.get("n_input_tokens") or 0) + int(row.get("n_output_tokens") or 0)
                for row in values
            ),
            "failures": sum(
                row.get("pass") is False or bool(row.get("exception_class"))
                for row in values
            ),
            "tool_calls": sum(int(row.get("weave_tool_call_count") or 0) for row in values),
            "turns": sum(int(row.get("weave_turn_count") or 0) for row in values),
            "recall_at_5": _average(_numbers(values, "recall_at_5")),
            "mrr": _average(_numbers(values, "mrr")),
        }
        evidence_id = f"E{index:03d}"
        payload["evidence_id"] = evidence_id
        aggregates.append(payload)
        evidence.append(
            EvidenceRef(
                id=evidence_id,
                kind="aggregate",
                label=" / ".join(key),
                row_ids=tuple(str(row.get("row_id")) for row in values),
                data=payload,
            )
        )
    return aggregates, evidence


def _scope(
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    warnings: Sequence[str],
) -> AnalysisScope:
    values = {
        "experiments": _unique(rows, "experiment_id"),
        "runs": _unique(rows, "run_id"),
        "tasks": _unique(rows, "task_name", fallback="task_id"),
        "models": _unique(rows, "model"),
        "variants": _unique(rows, "variant_id"),
        "sources": _unique(rows, "source"),
    }
    missing = tuple(
        metric
        for metric in metrics
        if not any(_metric_present(row, metric) for row in rows)
    )
    compatibility = {
        (
            str(row.get("workload_id") or "unknown"),
            str(row.get("model") or "unknown"),
        )
        for row in rows
    }
    scope_warnings = list(warnings)
    if len(compatibility) > 1:
        scope_warnings.append(
            "Scope contains different workload/model cohorts; results are stratified and should not be treated as one lift estimate."
        )
    sources = set(values["sources"])
    if any(row.get("weave_enriched") for row in rows):
        sources.add("weave")
    return AnalysisScope(
        experiments=values["experiments"],
        runs=values["runs"],
        rows=len(rows),
        tasks=values["tasks"],
        models=values["models"],
        variants=values["variants"],
        sources=tuple(sorted(sources)),
        missing_metrics=missing,
        warnings=tuple(scope_warnings),
    )


def _snapshot(rows: Sequence[dict[str, Any]], revision: str | None) -> AnalysisSnapshot:
    row_ids = tuple(sorted(str(row["row_id"]) for row in rows))
    digest = hashlib.sha256("\n".join(row_ids).encode()).hexdigest()
    return AnalysisSnapshot(
        id=f"snapshot-{digest[:12]}",
        digest=digest,
        created_at=datetime.now(UTC).isoformat(),
        catalog_revision=revision,
        row_ids=row_ids,
        rows=tuple(rows),
    )


def _merge_weave_record(
    local: dict[str, Any], remote: dict[str, Any] | None
) -> dict[str, Any]:
    if remote is None:
        return local
    return {
        **local,
        **remote,
        "weave_enriched": True,
    }


def _render_report(
    spec: AnalysisSpec,
    payload: Mapping[str, Any],
    evidence: Sequence[EvidenceRef],
    scope: AnalysisScope,
) -> str:
    valid = {item.id for item in evidence}
    claims = payload.get("claims") or []
    lines = [f"# {spec.title}", "", spec.question, "", "## Findings", ""]
    if not isinstance(claims, list) or not claims:
        raise ValueError("analysis report requires at least one evidence-backed claim")
    for claim in claims:
        if not isinstance(claim, dict) or not str(claim.get("text") or "").strip():
            raise ValueError("analysis claim must contain text")
        ids = [str(item) for item in claim.get("evidence_ids") or []]
        if not ids or any(item not in valid for item in ids):
            raise ValueError("every analysis claim must reference valid evidence ids")
        lines.append(f"- {str(claim['text']).strip()} {' '.join(f'[{item}]' for item in ids)}")
    conclusion = str(payload.get("conclusion") or "").strip()
    if conclusion:
        lines.extend(("", "## Conclusion", "", conclusion))
    if scope.warnings:
        lines.extend(("", "## Limitations", ""))
        lines.extend(f"- {warning}" for warning in scope.warnings)
    lines.extend(("", "## Evidence", ""))
    for item in evidence:
        location = item.local_path or item.weave_call_id or "normalized result rows"
        lines.append(f"- [{item.id}] {item.label}: `{location}`")
    return "\n".join(lines).rstrip() + "\n"


def _write_analysis(result: AnalysisResult) -> None:
    result.report_dir.mkdir(parents=True, exist_ok=True)
    (result.report_dir / "report.md").write_text(result.report)
    (result.report_dir / "analysis.json").write_text(
        json.dumps(
            {
                "spec": result.spec.to_dict(),
                "scope": asdict(result.scope),
                "snapshot": {key: value for key, value in asdict(result.snapshot).items() if key != "rows"},
                "aggregates": result.aggregates,
                "model": result.model,
                "provider": result.provider,
                "session_id": result.session_id,
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    (result.report_dir / "scope.json").write_text(
        json.dumps(asdict(result.snapshot), indent=2, sort_keys=True, default=str) + "\n"
    )
    with (result.report_dir / "evidence.jsonl").open("w") as handle:
        for item in result.evidence:
            handle.write(json.dumps(asdict(item), sort_keys=True, default=str) + "\n")


def _asset_draft(raw: Any) -> AssetDraft:
    if not isinstance(raw, dict):
        raise ValueError("asset draft must be an object")
    unknown = sorted(set(raw) - {"kind", "id", "title", "body"})
    if unknown:
        raise ValueError(f"unknown asset draft field(s): {', '.join(unknown)}")
    kind = str(raw.get("kind") or "")
    if kind not in {"prompt", "skill"}:
        raise ValueError("asset draft kind must be prompt or skill")
    item_id = validate_id(str(raw.get("id") or ""), kind=f"{kind} id")
    body = str(raw.get("body") or "").strip()
    if not body:
        raise ValueError(f"{kind} {item_id} body is required")
    return AssetDraft(
        kind=kind,  # type: ignore[arg-type]
        id=item_id,
        title=str(raw.get("title") or item_id),
        body=body + "\n",
    )


def _validate_experiment_references(
    experiment: ExperimentSpec,
    assets: Sequence[AssetDraft],
    repo_root: Path,
    env: Mapping[str, str],
) -> None:
    asset_ids = {(item.kind, item.id) for item in assets}
    allowed_harnesses = {"hermes", "openclaw", "claude-code", "codex"}
    unknown_harnesses = sorted(set(experiment.harnesses) - allowed_harnesses)
    if unknown_harnesses:
        raise ValueError(f"unknown harnesses: {', '.join(unknown_harnesses)}")
    for variant in experiment.variants:
        if variant.prompt_id and ("prompt", variant.prompt_id) not in asset_ids:
            get_prompt(variant.prompt_id, repo_root)
        for skill_id in variant.selected_skill_ids:
            if ("skill", skill_id) not in asset_ids:
                try:
                    get_skill(skill_id, repo_root)
                except FileNotFoundError:
                    load_skill_source(skill_id, repo_root)
        for integration in variant.integrations or experiment.integrations:
            load_integration(integration.id, repo_root)
        get_context_system(variant.context.system_id, repo_root)
    paths = [experiment.manifest]
    paths.extend(item.manifest for item in experiment.workloads if item.manifest)
    paths.extend(Path(item.dataset) for item in experiment.workloads if item.dataset)
    for path in paths:
        resolved = path if path.is_absolute() else repo_root / path
        if not resolved.is_file():
            raise FileNotFoundError(f"experiment data source not found: {path}")
    for model in (experiment.model, experiment.builder_model, experiment.judge_model):
        if not model:
            continue
        route = resolve_model_route(model, env)
        if not env.get(route.api_key_env, "").strip():
            raise RuntimeError(f"{route.api_key_env} is required for {model}")


def _client_factory(model: str, env: Mapping[str, str]) -> AssistantModelClient:
    return AssistantModelClient(model, env)


def _composer_instructions() -> str:
    return """You are Fugue's experiment composer. Build a complete, valid experiment from the user's request and repository catalog. Use only ids present in the catalog unless you include a new prompt or skill in assets. Preserve unspecified settings from the base experiment. Keep comparisons controlled: vary only what the user asks to vary. Use positive trials, task limits, and concurrency. Never include secrets. Call submit_experiment with the complete experiment, rationale, assumptions, warnings, and asset drafts. Do not claim that a draft has run."""


def _analyst_planner_instructions() -> str:
    return """You plan reproducible Fugue analyses. Select only catalog fields and ids that exist. Translate the question into exact filters, grouping dimensions, and deterministic metrics. Do not calculate results and do not invent experiments. Call submit_analysis_plan."""


def _analyst_report_instructions() -> str:
    return """You are Fugue's experiment analyst. Interpret only the supplied deterministic aggregates and evidence. Do not recalculate metrics, invent missing values, combine incompatible cohorts, or create a composite score. Every substantive claim must cite one or more provided evidence ids. Call submit_analysis_report with claims and a concise conclusion."""


def _experiment_submission_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "experiment": {"type": "object"},
            "assets": {"type": "array", "items": {"type": "object"}},
            "rationale": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["experiment", "rationale", "assumptions", "warnings"],
        "additionalProperties": False,
    }


def _analysis_plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "filters": {"type": "object", "additionalProperties": {"type": "string"}},
            "group_by": {"type": "array", "items": {"type": "string"}},
            "metrics": {"type": "array", "items": {"type": "string"}},
            "source": {"type": "string", "enum": ["local", "hybrid"]},
            "include_artifacts": {"type": "boolean"},
        },
        "required": ["id", "title", "filters", "group_by", "metrics"],
        "additionalProperties": False,
    }


def _analysis_report_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "evidence_ids"],
                    "additionalProperties": False,
                },
            },
            "conclusion": {"type": "string"},
        },
        "required": ["claims", "conclusion"],
        "additionalProperties": False,
    }


def _artifact_evidence(excerpt: ArtifactExcerpt, index: int) -> EvidenceRef:
    return EvidenceRef(
        id=f"E{index:03d}",
        kind="artifact",
        label=f"Artifact {excerpt.path}",
        local_path=excerpt.path,
        sha256=excerpt.sha256,
        data={"truncated": excerpt.truncated},
    )


def _scope_experiment_model(
    rows: Sequence[dict[str, Any]], repo_root: Path
) -> str | None:
    experiment_ids = _unique(rows, "experiment_id")
    if len(experiment_ids) != 1:
        return None
    try:
        return get_experiment(experiment_ids[0], repo_root).model
    except FileNotFoundError:
        return None


def _numbers(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _average(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _unique(
    rows: Sequence[Mapping[str, Any]], key: str, *, fallback: str | None = None
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(row.get(key) or (row.get(fallback) if fallback else "") or "unknown")
                for row in rows
            }
        )
    )


def _metric_present(row: Mapping[str, Any], metric: str) -> bool:
    aliases = {
        "pass_rate": "pass",
        "tokens": "n_input_tokens",
        "failures": "pass",
        "tool_calls": "weave_tool_call_count",
    }
    return row.get(aliases.get(metric, metric)) is not None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:64] or "analysis"
