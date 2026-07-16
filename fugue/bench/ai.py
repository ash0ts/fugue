from __future__ import annotations

import difflib
import hashlib
import json
import re
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
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
    AssistantUsage,
    select_assistant_model,
)
from fugue.bench.catalog import FILTER_FIELDS, ArtifactExcerpt, ExperimentCatalog
from fugue.bench.context import get_context_system, list_context_systems
from fugue.bench.evaluations import (
    EVALUATION_DIMENSIONS,
    EvaluationDraft,
    build_evaluation_draft,
    evaluation_asset_path,
    needs_evaluation_generation,
    source_catalog,
)
from fugue.bench.export import fetch_weave_summaries
from fugue.bench.integrations import list_integrations, load_integration
from fugue.bench.library import (
    ExperimentSpec,
    experiment_from_data,
    experiment_to_yaml,
    get_agent_preset,
    get_experiment,
    get_prompt,
    get_skill,
    list_agent_presets,
    list_experiments,
    list_prompts,
    list_skills,
    scorer_reference,
    validate_id,
)
from fugue.bench.manifest import load_manifest
from fugue.bench.scoring import (
    CandidateSelection,
    SelectionPolicy,
    build_treatment_selection_lock,
    select_candidate_configuration,
    write_treatment_selection_lock,
)
from fugue.bench.sources import list_skill_source_ids, load_skill_source
from fugue.model_plane import resolve_model_route, select_model, trace_project_slug
from fugue.redaction import redact_value

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
    kind: Literal[
        "prompt",
        "skill",
        "evaluation_cases",
        "evaluation_rubric",
        "evaluation_manifest",
    ]
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
    evaluation: EvaluationDraft | None = None


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
    selection: SelectionPolicy | None = None

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
    selection: CandidateSelection | None
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
    selection: CandidateSelection | None


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
        base_experiment: str | ExperimentSpec = "pilot",
        model: str | None = None,
        trace_content: str | None = None,
    ) -> ExperimentDraft:
        base = (
            base_experiment
            if isinstance(base_experiment, ExperimentSpec)
            else get_experiment(base_experiment, self.repo_root)
        )
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
            max_tokens=32_768,
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
                result = await self._complete_generated_evaluation(
                    result,
                    request=request,
                    client=client,
                    trace_content=trace_content or base.trace_content,
                )
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

    async def _complete_generated_evaluation(
        self,
        result: AssistantRunResult,
        *,
        request: str,
        client: AssistantModelClient,
        trace_content: str,
    ) -> AssistantRunResult:
        raw = result.payload
        experiment_raw = raw.get("experiment")
        if not isinstance(experiment_raw, dict):
            raise ValueError("submit_experiment requires an experiment object")
        assets = tuple(_asset_draft(item) for item in raw.get("assets") or [])
        experiment = experiment_from_data(experiment_raw)
        if not needs_evaluation_generation(experiment):
            return result
        _validate_experiment_references(
            experiment,
            assets,
            self.repo_root,
            self.operator.env,
            overlay=_asset_overlay(assets),
        )
        sources = source_catalog(
            experiment,
            self.repo_root,
            allow_mcp_io=True,
            draft_assets={(item.kind, item.id): item.body for item in assets},
        )
        base_messages = [
            AssistantMessage("system", _evaluation_generator_instructions()),
            AssistantMessage(
                "user",
                json.dumps(
                    {
                        "request": request,
                        "experiment": experiment.to_dict(),
                        "sources": [source.public() for source in sources],
                    },
                    sort_keys=True,
                    default=str,
                ),
            ),
        ]
        input_tokens = result.usage.input_tokens or 0
        output_tokens = result.usage.output_tokens or 0
        validation_error: Exception | None = None
        for _ in range(3):
            messages = list(base_messages)
            if validation_error is not None:
                messages.append(
                    AssistantMessage(
                        "user",
                        "The generated evaluation failed deterministic validation. "
                        f"Correct only the cases or rubric. Error: {validation_error}",
                    )
                )
            generator = AssistantAgent(
                client,
                role="composer",
                tools=(
                    AssistantTool(
                        "submit_evaluation",
                        "Submit the complete generated evaluation cases and rubric.",
                        _evaluation_submission_schema(),
                        terminal=True,
                    ),
                ),
                env=self.operator.env,
                trace_content=trace_content,
                max_rounds=2,
                max_tokens=32_768,
                attributes={
                    "fugue.ai.action": "generate_evaluation",
                    "fugue.ai.experiment_id": experiment.id,
                },
            )
            generated = await generator.run(messages)
            input_tokens += generated.usage.input_tokens or 0
            output_tokens += generated.usage.output_tokens or 0
            try:
                build_evaluation_draft(
                    generated.payload,
                    experiment,
                    generator_model=generated.model,
                    source_catalog=sources,
                    repo_root=self.repo_root,
                )
            except Exception as exc:
                validation_error = exc
                continue
            return AssistantRunResult(
                payload={**raw, "evaluation": generated.payload},
                messages=(*result.messages, *generated.messages),
                usage=AssistantUsage(input_tokens, output_tokens),
                model=result.model,
                provider=result.provider,
                session_id=result.session_id,
            )
        assert validation_error is not None
        raise ValueError(
            f"generated evaluation remained invalid: {validation_error}"
        )

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
                lambda _: self._catalog_summary(base, allow_mcp_io=True),
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
                "show_agent_preset",
                "Load one evidence-backed maintainer or operator agent preset.",
                {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
                lambda value: get_agent_preset(
                    str(value["id"]), self.repo_root
                ).to_dict(),
            ),
            AssistantTool(
                "submit_experiment",
                "Submit a complete Fugue experiment draft for deterministic validation.",
                _experiment_submission_schema(),
                terminal=True,
            ),
        )

    def _catalog_summary(
        self,
        base: ExperimentSpec,
        *,
        allow_mcp_io: bool = False,
    ) -> dict[str, Any]:
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
            "agent_presets": [
                get_agent_preset(item.id, self.repo_root).to_dict()
                for item in list_agent_presets(self.repo_root)
            ],
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
            "evaluation_sources": [
                item.public()
                for item in source_catalog(
                    base,
                    self.repo_root,
                    allow_mcp_io=allow_mcp_io,
                )
            ],
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
        asset_bodies = {(item.kind, item.id): item.body for item in assets}
        sources = source_catalog(
            experiment,
            self.repo_root,
            allow_mcp_io=True,
            draft_assets=asset_bodies,
        )
        evaluation: EvaluationDraft | None = None
        if raw.get("evaluation") is not None:
            if not experiment.judge_model:
                raise ValueError(
                    "generated evaluation rubrics require an explicit judge_model"
                )
            experiment, evaluation = build_evaluation_draft(
                raw["evaluation"],
                experiment,
                generator_model=result.model,
                source_catalog=sources,
                repo_root=self.repo_root,
            )
            assets = (
                *assets,
                *(
                    AssetDraft(
                        kind=item.kind,
                        id=item.suite_id,
                        title=item.path.name,
                        body=item.body,
                    )
                    for item in evaluation.files
                ),
            )
        elif needs_evaluation_generation(experiment):
            raise ValueError(
                "this experiment has no complete evaluation suite; submit an "
                "evaluation draft with grounded cases and a rubric"
            )
        overlay = _asset_overlay(assets)
        _validate_experiment_references(
            experiment,
            assets,
            self.repo_root,
            self.operator.env,
            overlay=overlay,
        )
        preview = self.operator.preview_experiment(experiment, asset_overlay=overlay)
        experiment_diff = "\n".join(
            difflib.unified_diff(
                experiment_to_yaml(base).splitlines(),
                experiment_to_yaml(experiment).splitlines(),
                fromfile=f"{base.id}.yaml",
                tofile=f"{experiment.id}.yaml",
                lineterm="",
            )
        )
        diff = _draft_diff(experiment_diff, assets, self.repo_root)
        return ExperimentDraft(
            experiment=experiment,
            assets=assets,
            evaluation=evaluation,
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
            spec = replace(spec, filters={**spec.filters, **filters})
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
        selection = _selection(rows, spec, snapshot.digest)
        if selection is not None:
            evidence.append(_selection_evidence(selection, len(evidence) + 1, rows))
        return AnalysisPreview(
            spec=spec,
            scope=scope,
            snapshot=snapshot,
            evidence=tuple(evidence),
            aggregates=tuple(aggregates),
            selection=selection,
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
        selection = _selection(rows, spec, snapshot.digest)
        if selection is not None:
            evidence.append(_selection_evidence(selection, len(evidence) + 1, rows))
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
                        "selection": selection.to_dict() if selection else None,
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
                            "selection": selection.to_dict() if selection else None,
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
            selection=selection,
            report=report,
            report_dir=report_dir,
            model=run.model,
            provider=run.provider,
            session_id=run.session_id,
            input_tokens=run.usage.input_tokens or 0,
            output_tokens=run.usage.output_tokens or 0,
        )
        _write_analysis(result, self.repo_root)
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
        selection=_selection_policy(raw.get("selection")),
    )


def _selection_policy(raw: Any) -> SelectionPolicy | None:
    if raw in (None, ""):
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("analysis selection must be a mapping")
    allowed = set(SelectionPolicy.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown analysis selection field(s): {', '.join(unknown)}")
    metric = str(raw.get("metric") or "pass_rate")
    if metric != "pass_rate":
        raise ValueError("analysis selection currently supports pass_rate only")
    confidence = float(raw.get("confidence", 0.95))
    margin = float(raw.get("noninferiority_margin", 0.05))
    samples = int(raw.get("bootstrap_samples", 2_000))
    if not 0.5 < confidence < 1.0:
        raise ValueError("analysis selection confidence must be between 0.5 and 1")
    if not 0 <= margin < 1.0:
        raise ValueError("analysis selection noninferiority_margin must be in [0, 1)")
    if samples < 100:
        raise ValueError("analysis selection bootstrap_samples must be at least 100")
    selection_unit = str(raw.get("selection_unit") or "candidate")
    if selection_unit not in {"candidate", "variant"}:
        raise ValueError("analysis selection_unit must be candidate or variant")
    tie_breakers = tuple(
        str(item)
        for item in raw.get("tie_breakers")
        or SelectionPolicy.__dataclass_fields__["tie_breakers"].default
    )
    supported_ties = {
        "cost_per_success",
        "median_wall_time_sec",
        "recoverable_error_rate",
        "localization_recall_at_10",
        "localization_mrr",
    }
    unknown_ties = sorted(set(tie_breakers) - supported_ties)
    if unknown_ties:
        raise ValueError(
            "unsupported analysis selection tie breaker(s): "
            + ", ".join(unknown_ties)
        )
    improvement_values = {
        "minimum_pass_rate_improvement": float(
            raw.get("minimum_pass_rate_improvement", 0.05)
        ),
        "minimum_cost_improvement": float(
            raw.get("minimum_cost_improvement", 0.15)
        ),
        "minimum_latency_improvement": float(
            raw.get("minimum_latency_improvement", 0.15)
        ),
    }
    if any(not 0 <= value < 1.0 for value in improvement_values.values()):
        raise ValueError("analysis selection improvement thresholds must be in [0, 1)")
    return SelectionPolicy(
        selection_unit=selection_unit,  # type: ignore[arg-type]
        baseline_variant_id=(
            str(raw["baseline_variant_id"])
            if raw.get("baseline_variant_id")
            else None
        ),
        required_examples=(
            int(raw["required_examples"]) if raw.get("required_examples") else None
        ),
        required_harnesses=tuple(
            str(value) for value in raw.get("required_harnesses") or ()
        ),
        require_agent_links=bool(raw.get("require_agent_links", False)),
        require_registration=bool(raw.get("require_registration", False)),
        metric=metric,
        confidence=confidence,
        noninferiority_margin=margin,
        require_complete_grid=bool(raw.get("require_complete_grid", True)),
        bootstrap_samples=samples,
        tie_breakers=tie_breakers,
        incumbent_candidate_id=(
            str(raw["incumbent_candidate_id"])
            if raw.get("incumbent_candidate_id")
            else None
        ),
        **improvement_values,
    )


def _selection(
    rows: Sequence[dict[str, Any]], spec: AnalysisSpec, snapshot_digest: str
) -> CandidateSelection | None:
    if spec.selection is None:
        return None
    return select_candidate_configuration(rows, spec.selection, seed=snapshot_digest)


def _selection_evidence(
    selection: CandidateSelection,
    index: int,
    rows: Sequence[dict[str, Any]],
) -> EvidenceRef:
    return EvidenceRef(
        id=f"E{index:03d}",
        kind="candidate_selection",
        label="Deterministic quality-first candidate selection",
        row_ids=tuple(str(row.get("row_id")) for row in rows),
        data=selection.to_dict(),
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


def _write_analysis(result: AnalysisResult, repo_root: Path) -> None:
    result.report_dir.mkdir(parents=True, exist_ok=True)
    (result.report_dir / "report.md").write_text(result.report)
    (result.report_dir / "analysis.json").write_text(
        json.dumps(
            {
                "spec": result.spec.to_dict(),
                "scope": asdict(result.scope),
                "snapshot": {key: value for key, value in asdict(result.snapshot).items() if key != "rows"},
                "aggregates": result.aggregates,
                "selection": result.selection.to_dict() if result.selection else None,
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
    _write_promotion_bundle(result, repo_root)


def _write_promotion_bundle(result: AnalysisResult, repo_root: Path) -> None:
    selection = result.selection
    if selection is None or selection.selected_candidate_id is None:
        return
    if result.spec.id == "repo-memory-discovery-selection":
        _write_memory_treatment_lock(result)
        return
    rows = [dict(row) for row in result.snapshot.rows]
    selected_rows = [
        row
        for row in rows
        if str(row.get("candidate_id")) == selection.selected_candidate_id
    ]
    if not selected_rows:
        return
    experiment_ids = _unique(selected_rows, "experiment_id")
    if len(experiment_ids) != 1:
        return
    experiment = get_experiment(experiment_ids[0], repo_root)
    role = next(
        (
            tag.split(":", 1)[1]
            for tag in experiment.tags
            if tag in {"role:maintainer", "role:operator"}
        ),
        None,
    )
    if role is None:
        return
    variant_id = str(selected_rows[0].get("variant_id") or "")
    variant = next((item for item in experiment.variants if item.id == variant_id), None)
    if variant is None:
        return
    score = next(
        item
        for item in selection.candidates
        if item.candidate_id == selection.selected_candidate_id
    )
    suite_id = next(
        (tag.split(":", 1)[1] for tag in experiment.tags if tag.startswith("suite:")),
        experiment.id,
    )
    suite_digest, base_commit = _suite_provenance(
        experiment, selected_rows, repo_root
    )
    campaign_tags = {
        str(tag)
        for row in selected_rows
        for tag in row.get("tags") or []
        if str(tag).startswith("campaign:")
    }
    campaign_tag = (
        next(iter(campaign_tags))
        if len(campaign_tags) == 1
        else str(result.spec.filters.get("tag") or "")
    )
    campaign_id = (
        campaign_tag.split(":", 1)[1]
        if campaign_tag.startswith("campaign:")
        else result.snapshot.id
    )
    output = repo_root / "reports" / "self-eval" / validate_id(
        _slug(campaign_id), kind="campaign id"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "pass_rate": score.pass_rate,
        "cost_per_success": score.cost_per_success,
        "median_wall_time_sec": score.median_wall_time_sec,
        "recoverable_error_rate": score.recoverable_error_rate,
        "confidence_low": score.confidence_low,
        "confidence_high": score.confidence_high,
    }
    proposal = {
        "decision": selection.decision,
        "reason": selection.reason,
        "role": role,
        "suite_id": suite_id,
        "suite_digest": suite_digest,
        "base_commit": base_commit,
        "analysis_id": result.spec.id,
        "analysis_snapshot": result.snapshot.digest,
        "run_ids": list(result.scope.runs),
        "best_candidate_id": selection.best_candidate_id,
        "selected_candidate_id": selection.selected_candidate_id,
        "policy": asdict(selection.policy),
        "metrics": metrics,
        "candidates": [asdict(item) for item in selection.candidates],
    }
    (output / "promotion.json").write_text(
        json.dumps(proposal, indent=2, sort_keys=True, default=str) + "\n"
    )
    (output / "promotion.md").write_text(
        "\n".join(
            [
                f"# Fugue {role.title()} Agent Promotion",
                "",
                f"- Decision: `{selection.decision}`",
                f"- Candidate: `{selection.selected_candidate_id}`",
                f"- Reason: {selection.reason}",
                f"- Suite: `{suite_id}` (`{suite_digest[:12]}`)",
                f"- Source: `{base_commit}`",
                f"- Analysis snapshot: `{result.snapshot.digest}`",
                "",
                "This is a review artifact. It does not change tracked defaults or open a PR.",
                "",
            ]
        )
    )
    candidate = {
        "id": f"fugue-{role}-recommended",
        "title": f"Recommended Fugue {role}",
        "role": role,
        "base_experiment_id": experiment.id,
        "candidate": {
            "harness": selected_rows[0].get("harness"),
            "model": selected_rows[0].get("model"),
            "prompt_id": variant.prompt_id,
            "skills": variant.skills,
            "context": asdict(variant.context),
            "integrations": [asdict(item) for item in variant.integrations],
            "agent_kwargs": variant.agent_kwargs,
            "agent_env": variant.agent_env,
            "environment": variant.environment,
            "verifier": variant.verifier,
            "retry": variant.retry,
            "artifacts": variant.artifacts,
        },
        "evidence": {
            "suite_id": suite_id,
            "suite_digest": suite_digest,
            "base_commit": base_commit,
            "run_ids": list(result.scope.runs),
            "analysis_snapshot": result.snapshot.digest,
            "metrics": metrics,
        },
    }
    (output / "candidate-preset.yaml").write_text(
        yaml.safe_dump(redact_value(candidate), sort_keys=False)
    )


def _write_memory_treatment_lock(result: AnalysisResult) -> None:
    selection = result.selection
    assert selection is not None
    if any(not candidate.eligible for candidate in selection.candidates):
        return
    eligible = [candidate for candidate in selection.candidates if candidate.eligible]
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            -(candidate.paired_pass_rate_delta or 0.0),
            -(
                candidate.localization_recall_at_10
                if candidate.localization_recall_at_10 is not None
                else -1.0
            ),
            -(
                candidate.localization_mrr
                if candidate.localization_mrr is not None
                else -1.0
            ),
            candidate.recoverable_error_rate
            if candidate.recoverable_error_rate is not None
            else float("inf"),
            candidate.cost_per_success
            if candidate.cost_per_success is not None
            else float("inf"),
            candidate.candidate_id,
        ),
    )
    if len(ranked) < 3:
        return
    rows = [dict(row) for row in result.snapshot.rows]
    calibration = {
        str(row.get("run_snapshot_sha256") or "")
        for row in rows
        if row.get("workload_id") == "hard-calibration"
    }
    discovery = {
        str(row.get("run_snapshot_sha256") or "")
        for row in rows
        if row.get("workload_id") == "hard-discovery"
    }
    source_commits = {str(row.get("source_commit") or "") for row in rows}
    if len(calibration) != 1 or len(discovery) != 1 or len(source_commits) != 1:
        return
    rankings = tuple(
        {
            "rank": index,
            "variant_id": candidate.candidate_id,
            "eligible": candidate.eligible,
            "paired_pass_rate_delta": candidate.paired_pass_rate_delta,
            "localization_recall_at_10": candidate.localization_recall_at_10,
            "localization_mrr": candidate.localization_mrr,
            "recoverable_error_rate": candidate.recoverable_error_rate,
            "cost_per_success": candidate.cost_per_success,
            "reasons": list(candidate.reasons),
        }
        for index, candidate in enumerate(ranked, start=1)
    )
    lock = build_treatment_selection_lock(
        source_commit=next(iter(source_commits)),
        calibration_snapshot_sha256=next(iter(calibration)),
        discovery_snapshot_sha256=next(iter(discovery)),
        rankings=rankings,
        selected_variants=[candidate.candidate_id for candidate in ranked[:3]],
    )
    write_treatment_selection_lock(
        result.report_dir / "treatment-selection-lock.json", lock
    )


def _suite_provenance(
    experiment: ExperimentSpec,
    rows: Sequence[dict[str, Any]],
    repo_root: Path,
) -> tuple[str, str]:
    workload_ids = {str(row.get("workload_id") or "") for row in rows}
    manifest_paths = {
        item.manifest
        for item in experiment.workloads
        if item.id in workload_ids and item.manifest is not None
    } or {experiment.manifest}
    digest = hashlib.sha256()
    commits: set[str] = set()
    dataset_roots: set[Path] = set()
    for value in sorted(manifest_paths, key=lambda item: item.as_posix()):
        path = value if value.is_absolute() else repo_root / value
        digest.update(path.relative_to(repo_root).as_posix().encode())
        digest.update(path.read_bytes())
        manifest = load_manifest(path)
        commits.update(task.base_commit for task in manifest.tasks if task.base_commit)
        if manifest.dataset.path:
            dataset_roots.add(
                manifest.dataset.path
                if manifest.dataset.path.is_absolute()
                else repo_root / manifest.dataset.path
            )
    for root in sorted(dataset_roots, key=lambda item: item.as_posix()):
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest.update(path.relative_to(repo_root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest(), next(iter(commits)) if len(commits) == 1 else "mixed"


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
    *,
    overlay: Mapping[str, str] | None = None,
) -> None:
    asset_ids = {(item.kind, item.id) for item in assets}
    allowed_harnesses = {"hermes", "openclaw", "claude-code", "codex"}
    unknown_harnesses = sorted(set(experiment.harnesses) - allowed_harnesses)
    if unknown_harnesses:
        raise ValueError(f"unknown harnesses: {', '.join(unknown_harnesses)}")
    for variant in experiment.variants:
        if variant.prompt_id and ("prompt", variant.prompt_id) not in asset_ids:
            get_prompt(variant.prompt_id, repo_root)
        for skill_id in variant.skills:
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
    paths.extend(
        Path(scorer_reference(scorer))
        for item in experiment.workloads
        for scorer in item.scorers
        if not scorer_reference(scorer).startswith("builtin:")
    )
    virtual_paths = set(overlay or {})
    for path in paths:
        if path.as_posix() in virtual_paths:
            continue
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
    return """You are Fugue's experiment composer. Build a complete, valid experiment from the user's request and repository catalog. Use only ids present in the catalog unless you include a new prompt or skill in assets. Never copy an existing catalog asset into assets. Evidence-backed agent presets are optional starting points and must be applied explicitly when the request calls for that role. Preserve unspecified settings from the base experiment. Every authored variant must declare context as a mapping with both system_id and delivery, including no-context variants. Keep comparisons controlled: vary only what the user asks to vary, and include a baseline variant that omits the tested capability. Use positive trials, task limits, and concurrency. When an evaluation is missing, configure evaluation_generation with the exact suite id, workload id, size, and typed sources. Do not generate cases or a rubric in this stage; Fugue runs that explicit bounded generation after the experiment plan validates. Generated rubrics require an explicit judge_model. Never include secrets. Call submit_experiment with the complete experiment, optional new prompt/skill asset drafts, rationale, assumptions, and warnings. Do not claim that a draft has run."""


def _evaluation_generator_instructions() -> str:
    return """Generate only the evaluation suite configured by the supplied experiment. Call submit_evaluation with exactly evaluation_generation.size concise grounded cases and one rubric. Do not repeat source content or the experiment inside instructions or assertions. Every case must cite supplied source ids and contain at least one expected fact, tool assertion, or artifact assertion. Cover easy, boundary, failure, and integration behavior across the applicable families. Use only the five supported rubric dimensions, separate 0..1 scores, and a default 0.7 threshold. Never invent sources, include secrets, or claim the suite has run."""


def _analyst_planner_instructions() -> str:
    return """You plan reproducible Fugue analyses. Select only catalog fields and ids that exist. Translate the question into exact filters, grouping dimensions, and deterministic metrics. Do not calculate results and do not invent experiments. Call submit_analysis_plan."""


def _analyst_report_instructions() -> str:
    return """You are Fugue's experiment analyst. Interpret only the supplied deterministic aggregates and evidence. Do not recalculate metrics, invent missing values, combine incompatible cohorts, or create a composite score. Every substantive claim must cite one or more provided evidence ids. Call submit_analysis_report with claims and a concise conclusion."""


def _experiment_submission_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "experiment": {"type": "object"},
            "assets": {
                "type": "array",
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["prompt", "skill"]},
                        "id": {"type": "string", "maxLength": 128},
                        "title": {"type": "string", "maxLength": 256},
                        "body": {"type": "string", "maxLength": 64_000},
                    },
                    "required": ["kind", "id", "title", "body"],
                    "additionalProperties": False,
                },
            },
            "rationale": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["experiment", "rationale", "assumptions", "warnings"],
        "additionalProperties": False,
    }


def _evaluation_submission_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "suite_id": {"type": "string", "maxLength": 128},
            "cases": {
                "type": "array",
                "maxItems": 64,
                "items": _evaluation_case_submission_schema(),
            },
            "rubric": _evaluation_rubric_submission_schema(),
        },
        "required": ["suite_id", "cases", "rubric"],
        "additionalProperties": False,
    }


def _evaluation_case_submission_schema() -> dict[str, Any]:
    short_text = {"type": "string", "maxLength": 500}
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "maxLength": 128},
            "instruction": {"type": "string", "maxLength": 2_000},
            "family": {
                "type": "string",
                "enum": ["prompt", "skill", "mcp", "agent", "mixed"],
            },
            "source_refs": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {"type": "string", "maxLength": 256},
            },
            "attachments": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "maxLength": 512},
                        "target": {"type": "string", "maxLength": 512},
                        "sha256": {"type": "string", "maxLength": 64},
                    },
                    "required": ["path", "target", "sha256"],
                    "additionalProperties": False,
                },
            },
            "expected": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "maxItems": 8,
                        "items": short_text,
                    },
                    "tool_calls": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "server": {"type": "string", "maxLength": 128},
                                "tool": {"type": "string", "maxLength": 128},
                                "arguments_subset": {
                                    "type": "object",
                                    "maxProperties": 16,
                                },
                            },
                            "required": ["tool"],
                            "additionalProperties": False,
                        },
                    },
                    "artifacts": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "maxLength": 512},
                                "checks": {
                                    "type": "array",
                                    "maxItems": 3,
                                    "items": {
                                        "type": "string",
                                        "enum": ["exists", "nonempty", "json"],
                                    },
                                },
                            },
                            "required": ["path", "checks"],
                            "additionalProperties": False,
                        },
                    },
                    "reference_answer": {"type": "string", "maxLength": 2_000},
                },
                "additionalProperties": False,
            },
            "scorer_dimensions": {
                "type": "array",
                "maxItems": len(EVALUATION_DIMENSIONS),
                "items": {
                    "type": "string",
                    "enum": sorted(EVALUATION_DIMENSIONS),
                },
            },
            "tags": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": {"type": "string", "maxLength": 128},
            },
            "turns": {
                "type": "array",
                "maxItems": 4,
                "items": {"type": "string", "maxLength": 1_000},
            },
        },
        "required": ["id", "instruction", "family", "source_refs", "expected", "tags"],
        "additionalProperties": False,
    }


def _evaluation_rubric_submission_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "dimensions": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(EVALUATION_DIMENSIONS),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "enum": sorted(EVALUATION_DIMENSIONS),
                        },
                        "kind": {"type": "string", "enum": ["llm_judge"]},
                        "criterion": {"type": "string", "maxLength": 1_000},
                        "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string", "maxLength": 256},
                        },
                    },
                    "required": ["id", "kind", "criterion", "threshold", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["dimensions"],
        "additionalProperties": False,
    }


def _asset_overlay(assets: Sequence[AssetDraft]) -> dict[str, str]:
    values: dict[str, str] = {}
    for asset in assets:
        if asset.kind == "prompt":
            path = Path("configs/fugue/prompts") / f"{asset.id}.md"
        elif asset.kind == "skill":
            path = Path("configs/fugue/skills") / asset.id / "SKILL.md"
        else:
            path = evaluation_asset_path(asset.kind, asset.id)
        values[path.as_posix()] = asset.body
    return values


def _draft_diff(
    experiment_diff: str,
    assets: Sequence[AssetDraft],
    repo_root: Path,
) -> str:
    sections = [experiment_diff] if experiment_diff else []
    for path_text, body in _asset_overlay(assets).items():
        path = repo_root / path_text
        before = path.read_text() if path.is_file() else ""
        section = "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                body.splitlines(),
                fromfile=path_text if before else "/dev/null",
                tofile=path_text,
                lineterm="",
            )
        )
        if section:
            sections.append(section)
    return "\n\n".join(sections)


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
            "selection": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["pass_rate"]},
                    "confidence": {"type": "number"},
                    "noninferiority_margin": {"type": "number"},
                    "require_complete_grid": {"type": "boolean"},
                    "bootstrap_samples": {"type": "integer"},
                    "tie_breakers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "incumbent_candidate_id": {"type": "string"},
                    "minimum_pass_rate_improvement": {"type": "number"},
                    "minimum_cost_improvement": {"type": "number"},
                    "minimum_latency_improvement": {"type": "number"},
                },
                "additionalProperties": False,
            },
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
