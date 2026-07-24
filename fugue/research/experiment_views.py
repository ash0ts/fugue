from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from fugue.bench.candidates import stable_digest

EXPERIMENT_VIEW_SCHEMA_VERSION = 1
EXPERIMENT_VIEW_CELL_LIMIT = 256

ExperimentViewKind = Literal["design", "progress", "evaluation"]
ExecutionStatus = Literal[
    "queued",
    "preparing",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "not_applicable",
]
OutcomeStatus = Literal["pending", "passed", "failed", "unavailable", "not_applicable"]
EvidenceStatus = Literal["pending", "reconciled", "missing", "not_applicable"]
SummaryStatus = Literal["passed", "failed", "unavailable", "not_applicable"]
ScoreStatus = Literal[
    "passed", "failed", "observed", "unavailable", "not_applicable"
]

_EXECUTION_STATES = {
    "queued",
    "preparing",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "not_applicable",
}
_OUTCOME_STATES = {"pending", "passed", "failed", "unavailable", "not_applicable"}
_EVIDENCE_STATES = {"pending", "reconciled", "missing", "not_applicable"}
_VIEW_KINDS = {"design", "progress", "evaluation"}
_SAFE_BEHAVIORAL_MEASURES = (
    "context_invoked",
    "localization_recall_at_5",
    "localization_recall_at_10",
    "localization_mrr",
    "relevant_retrieval_open_rate",
    "relevant_retrieval_change_rate",
    "off_target_change_only",
    "premature_completion",
    "prompt_injection_action_gate_allowed",
    "prompt_injection_action_gate_blocked",
    "prompt_injection_attack_encountered",
    "prompt_injection_compromised",
    "prompt_injection_evidence_preserved",
    "prompt_injection_false_positive_refusal",
    "prompt_injection_incorrect",
    "prompt_injection_safe_and_useful",
    "prompt_injection_safe_but_failed_or_refused",
    "prompt_injection_sensitive_action_attempted",
    "prompt_injection_task_complete",
    "tool_calls",
    "turns",
    "input_tokens",
    "output_tokens",
    "wall_time_sec",
)


@dataclass(frozen=True)
class ExperimentDescriptorV1:
    id: str
    label: str
    digest: str | None = None
    details: dict[str, str | int | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class ExperimentFactorV1:
    name: str
    levels: tuple[str, ...]
    label: str | None = None
    level_labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class ExperimentOutcomeSummaryV1:
    id: str
    label: str
    status: SummaryStatus
    passed: int | None = None
    total: int | None = None
    unavailable: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentTaskDesignV1:
    title: str
    summary: str
    interaction_mode: str | None = None
    tools: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    evidence_links: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentPromptDesignV1:
    base_instruction_summary: str
    treatment_summaries: dict[str, str] = field(default_factory=dict)
    evidence_links: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentScoreDefinitionV1:
    id: str
    label: str
    description: str | None = None
    source_key: str | None = None
    target: str | float | int | bool | None = None
    primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentScorerDesignV1:
    id: str
    label: str
    kind: Literal["benchmark", "deterministic", "criteria", "llm_judge"]
    description: str
    required: bool
    threshold: float | None = None
    aggregation: str | None = None
    evidence_inputs: tuple[str, ...] = ()
    revision: str | None = None
    model: str | None = None
    rubric_summary: str | None = None
    blind_fields: tuple[str, ...] = ()
    dimensions: tuple[ExperimentScoreDefinitionV1, ...] = ()
    evidence_links: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentEvaluationDesignV1:
    pass_rule: str
    scorers: tuple[ExperimentScorerDesignV1, ...]
    llm_judge_used: bool

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentScoreResultV1:
    id: str
    label: str
    status: ScoreStatus
    value: str | float | int | bool | None = None
    scorer_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentScoreSummaryV1:
    id: str
    label: str
    observed: int
    passed: int | None = None
    failed: int | None = None
    unavailable: int = 0
    mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentCellViewV1:
    cell_id: str
    task_label: str
    factor_levels: dict[str, str]
    attempt: int
    execution_status: ExecutionStatus
    task_outcome: OutcomeStatus
    evaluation_status: OutcomeStatus
    evidence_status: EvidenceStatus
    reason_code: str | None = None
    cost_usd: float | None = None
    latency_sec: float | None = None
    evidence_links: tuple[dict[str, str], ...] = ()
    measures: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    scores: tuple[ExperimentScoreResultV1, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ExperimentViewV1:
    schema_version: int
    kind: ExperimentViewKind
    research_label: str | None = None
    study_label: str | None = None
    question: str | None = None
    hypothesis: str | None = None
    context: str | None = None
    observation: str | None = None
    rationale: str | None = None
    alternative_explanations: tuple[str, ...] = ()
    success_definition: str | None = None
    task_design: ExperimentTaskDesignV1 | None = None
    prompt_design: ExperimentPromptDesignV1 | None = None
    evaluation_design: ExperimentEvaluationDesignV1 | None = None
    source_cohort: ExperimentDescriptorV1 | None = None
    fixed_conditions: tuple[ExperimentFactorV1, ...] = ()
    varied_factors: tuple[ExperimentFactorV1, ...] = ()
    measured_outcomes: tuple[str, ...] = ()
    taskset: ExperimentDescriptorV1 | None = None
    harnesses: tuple[ExperimentDescriptorV1, ...] = ()
    runtime: ExperimentDescriptorV1 | None = None
    matrix_size: int = 0
    preview_digest: str | None = None
    approval_state: str | None = None
    cell_limit: int | None = None
    reserved_cost_usd: float | None = None
    phase: str | None = None
    completed_cells: int | None = None
    observed_cost_usd: float | None = None
    state_counts: dict[str, int] = field(default_factory=dict)
    cells: tuple[ExperimentCellViewV1, ...] = ()
    omitted_cells: int = 0
    infrastructure_health: str | None = None
    arm_totals: tuple[dict[str, Any], ...] = ()
    aligned_comparisons: tuple[dict[str, Any], ...] = ()
    behavioral_measures: dict[str, Any] = field(default_factory=dict)
    outcome_summaries: tuple[ExperimentOutcomeSummaryV1, ...] = ()
    score_summaries: tuple[ExperimentScoreSummaryV1, ...] = ()
    evidence_eligible: bool | None = None
    limitations: tuple[str, ...] = ()
    evidence_links: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


def experiment_view_from_dict(raw: Mapping[str, Any]) -> ExperimentViewV1:
    _reject_unknown(
        raw,
        {item.name for item in ExperimentViewV1.__dataclass_fields__.values()},
        "experiment view",
    )
    schema_version = _positive_int(raw.get("schema_version"), "schema_version")
    if schema_version != EXPERIMENT_VIEW_SCHEMA_VERSION:
        raise ValueError("unsupported experiment view schema")
    kind = _text(raw.get("kind"), "kind", 40)
    if kind not in _VIEW_KINDS:
        raise ValueError("unknown experiment view kind")
    fixed = tuple(
        _factor(item, "fixed_conditions")
        for item in _sequence(raw.get("fixed_conditions"), "fixed_conditions")
    )
    varied = tuple(
        _factor(item, "varied_factors")
        for item in _sequence(raw.get("varied_factors"), "varied_factors")
    )
    cells = tuple(_cell(item) for item in _sequence(raw.get("cells"), "cells"))
    if len(cells) > EXPERIMENT_VIEW_CELL_LIMIT:
        raise ValueError("experiment view exceeds the public cell limit")
    view = ExperimentViewV1(
        schema_version=schema_version,
        kind=kind,  # type: ignore[arg-type]
        research_label=_optional_text(raw.get("research_label"), "research_label", 300),
        study_label=_optional_text(raw.get("study_label"), "study_label", 300),
        question=_optional_text(raw.get("question"), "question", 2000),
        hypothesis=_optional_text(raw.get("hypothesis"), "hypothesis", 2000),
        context=_optional_text(raw.get("context"), "context", 4000),
        observation=_optional_text(raw.get("observation"), "observation", 4000),
        rationale=_optional_text(raw.get("rationale"), "rationale", 4000),
        alternative_explanations=tuple(
            _text(item, "alternative explanation", 1000)
            for item in _sequence(
                raw.get("alternative_explanations"), "alternative_explanations"
            )
        ),
        success_definition=_optional_text(
            raw.get("success_definition"), "success_definition", 4000
        ),
        task_design=_optional_task_design(raw.get("task_design")),
        prompt_design=_optional_prompt_design(raw.get("prompt_design")),
        evaluation_design=_optional_evaluation_design(
            raw.get("evaluation_design")
        ),
        source_cohort=_optional_descriptor(raw.get("source_cohort"), "source_cohort"),
        fixed_conditions=fixed,
        varied_factors=varied,
        measured_outcomes=tuple(
            _text(item, "measured_outcome", 200)
            for item in _sequence(raw.get("measured_outcomes"), "measured_outcomes")
        ),
        taskset=_optional_descriptor(raw.get("taskset"), "taskset"),
        harnesses=tuple(
            _descriptor(item, "harnesses")
            for item in _sequence(raw.get("harnesses"), "harnesses")
        ),
        runtime=_optional_descriptor(raw.get("runtime"), "runtime"),
        matrix_size=_non_negative_int(raw.get("matrix_size", 0), "matrix_size"),
        preview_digest=_optional_digest(raw.get("preview_digest"), "preview_digest"),
        approval_state=_optional_text(raw.get("approval_state"), "approval_state", 80),
        cell_limit=_optional_non_negative_int(raw.get("cell_limit"), "cell_limit"),
        reserved_cost_usd=_optional_cost(
            raw.get("reserved_cost_usd"), "reserved_cost_usd"
        ),
        phase=_optional_text(raw.get("phase"), "phase", 80),
        completed_cells=_optional_non_negative_int(
            raw.get("completed_cells"), "completed_cells"
        ),
        observed_cost_usd=_optional_cost(
            raw.get("observed_cost_usd"), "observed_cost_usd"
        ),
        state_counts=_count_mapping(raw.get("state_counts"), "state_counts"),
        cells=cells,
        omitted_cells=_non_negative_int(raw.get("omitted_cells", 0), "omitted_cells"),
        infrastructure_health=_optional_text(
            raw.get("infrastructure_health"), "infrastructure_health", 80
        ),
        arm_totals=tuple(
            _arm_total(item) for item in _sequence(raw.get("arm_totals"), "arm_totals")
        ),
        aligned_comparisons=tuple(
            _comparison(item)
            for item in _sequence(raw.get("aligned_comparisons"), "aligned_comparisons")
        ),
        behavioral_measures=_measure_mapping(
            raw.get("behavioral_measures"), "behavioral_measures"
        ),
        outcome_summaries=tuple(
            _outcome_summary(item)
            for item in _sequence(raw.get("outcome_summaries"), "outcome_summaries")
        ),
        score_summaries=tuple(
            _score_summary(item)
            for item in _sequence(raw.get("score_summaries"), "score_summaries")
        ),
        evidence_eligible=_optional_bool(
            raw.get("evidence_eligible"), "evidence_eligible"
        ),
        limitations=tuple(
            _text(item, "limitation", 1000)
            for item in _sequence(raw.get("limitations"), "limitations")
        ),
        evidence_links=_evidence_links(raw.get("evidence_links")),
    )
    _validate_view_shape(view)
    return view


def build_design_view(
    preview: Mapping[str, Any], *, approval_state: str = "awaiting_approval"
) -> ExperimentViewV1:
    draft = _mapping(preview.get("draft"), "draft")
    plan = _mapping_or_empty(preview.get("plan_receipt"))
    cells = [
        item
        for item in _sequence(plan.get("cells"), "plan_receipt.cells")
        if isinstance(item, Mapping)
    ]
    fixed_names = tuple(str(item) for item in draft.get("fixed_dimensions") or ())
    varied_names = tuple(str(item) for item in draft.get("varied_dimensions") or ())
    labels = _display_labels(draft.get("display_labels"))
    fixed = tuple(
        ExperimentFactorV1(
            name=_dimension_label(name),
            levels=_levels_for(name, draft, cells),
            label=labels.get(_dimension_label(name), labels.get(name)),
            level_labels=_factor_level_labels(name, draft, cells, labels),
        )
        for name in fixed_names
    )
    varied = tuple(
        ExperimentFactorV1(
            name=_dimension_label(name),
            levels=_levels_for(name, draft, cells),
            label=labels.get(_dimension_label(name), labels.get(name)),
            level_labels=_factor_level_labels(name, draft, cells, labels),
        )
        for name in varied_names
    )
    matrix_size = int(preview.get("estimated_cells") or len(cells))
    display_cells = tuple(
        _planned_cell(item, tuple(factor.name for factor in varied))
        for item in cells[:EXPERIMENT_VIEW_CELL_LIMIT]
    )
    task_count = draft.get("n_tasks")
    if task_count is None:
        task_count = len(
            {str(item.get("task_id") or "") for item in cells if item.get("task_id")}
        )
    taskset_digest = str(
        draft.get("task_suite_digest")
        or _mapping_or_empty(preview.get("task_suite_preview")).get("preview_digest")
        or ""
    )
    taskset = ExperimentDescriptorV1(
        id=str(
            draft.get("task_suite_digest")
            or draft.get("preset_id")
            or "registered-taskset"
        ),
        label=_taskset_label(draft, task_count),
        digest=taskset_digest or None,
        details={"task_count": int(task_count or 0)},
    )
    preview_digest = str(preview.get("preview_digest") or "")
    harness_ids = _ordered_values(
        [str(item) for item in draft.get("harnesses") or ()]
        + [str(item.get("harness") or "") for item in cells]
    )
    recipe = _mapping_or_empty(draft.get("task_recipe_preview"))
    provenance = _mapping_or_empty(recipe.get("provenance"))
    selected_calls = [
        item
        for item in provenance.get("selected_call_ids") or ()
        if isinstance(item, str) and item
    ]
    source_cohort = None
    if provenance.get("trace_audit_id") and selected_calls:
        source_cohort = ExperimentDescriptorV1(
            id=str(provenance["trace_audit_id"]),
            label=(
                f"{len(selected_calls)} reviewed Weave "
                f"call{'s' if len(selected_calls) != 1 else ''}"
            ),
            digest=(
                str(provenance["trace_audit_digest"])
                if provenance.get("trace_audit_digest")
                else None
            ),
            details={
                "call_count": len(selected_calls),
                "system": "weave",
            },
        )
    context = str(draft.get("decision_rationale") or "").strip() or None
    research_view = _mapping_or_empty(draft.get("research_view"))
    task_design = _research_task_design(research_view, recipe)
    prompt_design = _research_prompt_design(research_view)
    evaluation_design = _research_evaluation_design(research_view)
    if task_design is not None and not task_design.evidence_links:
        task_reference = {
            "system": "fugue",
            "kind": "task_definition",
            "ref": taskset.id,
        }
        if taskset.digest:
            task_reference["digest"] = taskset.digest
        task_design = ExperimentTaskDesignV1(
            title=task_design.title,
            summary=task_design.summary,
            interaction_mode=task_design.interaction_mode,
            tools=task_design.tools,
            resources=task_design.resources,
            evidence_links=(task_reference,),
        )
    if prompt_design is not None and not prompt_design.evidence_links and preview_digest:
        prompt_design = ExperimentPromptDesignV1(
            base_instruction_summary=prompt_design.base_instruction_summary,
            treatment_summaries=prompt_design.treatment_summaries,
            evidence_links=(
                {
                    "system": "fugue",
                    "kind": "prompt_design",
                    "ref": preview_digest,
                    "digest": preview_digest,
                },
            ),
        )
    return experiment_view_from_dict(
        ExperimentViewV1(
            schema_version=EXPERIMENT_VIEW_SCHEMA_VERSION,
            kind="design",
            research_label=labels.get("research"),
            study_label=labels.get("study"),
            question=str(draft.get("question") or draft.get("research_question") or ""),
            hypothesis=str(draft.get("hypothesis") or ""),
            context=context,
            observation=str(research_view.get("observation") or "").strip() or None,
            rationale=str(research_view.get("rationale") or "").strip() or context,
            alternative_explanations=tuple(
                str(item)
                for item in research_view.get("alternative_explanations") or ()
                if str(item).strip()
            ),
            success_definition=(
                str(research_view.get("success_definition") or "").strip() or None
            ),
            task_design=task_design,
            prompt_design=prompt_design,
            evaluation_design=evaluation_design,
            source_cohort=source_cohort,
            fixed_conditions=fixed,
            varied_factors=varied,
            measured_outcomes=tuple(
                str(item) for item in draft.get("measured_dimensions") or ()
            ),
            taskset=taskset,
            harnesses=tuple(
                ExperimentDescriptorV1(
                    id=value, label=labels.get(value, _humanize(value))
                )
                for value in harness_ids
            ),
            runtime=ExperimentDescriptorV1(
                id="harbor",
                label="Harbor isolated runtime",
                details={"locked_before_execution": True},
            ),
            matrix_size=matrix_size,
            preview_digest=preview_digest or None,
            approval_state=approval_state,
            cell_limit=matrix_size,
            reserved_cost_usd=float(preview.get("estimated_cost_usd") or 0.0),
            cells=display_cells,
            omitted_cells=max(0, len(cells) - len(display_cells)),
        ).to_dict()
    )


def _research_task_design(
    research_view: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> ExperimentTaskDesignV1 | None:
    title = str(research_view.get("task_title") or "").strip()
    summary = str(research_view.get("task_summary") or "").strip()
    if not title or not summary:
        return None
    links: list[dict[str, str]] = []
    preview_digest = str(recipe.get("preview_digest") or "")
    recipe_id = str(recipe.get("recipe_id") or "")
    if recipe_id and preview_digest:
        links.append(
            {
                "system": "fugue",
                "kind": "task_definition",
                "ref": recipe_id,
                "digest": preview_digest,
            }
        )
    return ExperimentTaskDesignV1(
        title=title,
        summary=summary,
        interaction_mode=(
            str(research_view.get("interaction_mode") or "").strip() or None
        ),
        tools=tuple(
            str(item)
            for item in research_view.get("tools") or ()
            if str(item).strip()
        ),
        resources=tuple(
            str(item)
            for item in research_view.get("resources") or ()
            if str(item).strip()
        ),
        evidence_links=tuple(links),
    )


def _research_prompt_design(
    research_view: Mapping[str, Any],
) -> ExperimentPromptDesignV1 | None:
    summary = str(research_view.get("base_instruction_summary") or "").strip()
    treatments = {
        str(key): str(value)
        for key, value in _mapping_or_empty(
            research_view.get("treatment_summaries")
        ).items()
        if str(key).strip() and str(value).strip()
    }
    if not summary and not treatments:
        return None
    return ExperimentPromptDesignV1(
        base_instruction_summary=summary or "No additional base instruction summary.",
        treatment_summaries=treatments,
    )


def _research_evaluation_design(
    research_view: Mapping[str, Any],
) -> ExperimentEvaluationDesignV1 | None:
    raw_scorers = [
        item
        for item in research_view.get("scorers") or ()
        if isinstance(item, Mapping)
    ]
    pass_rule = str(research_view.get("pass_rule") or "").strip()
    if not raw_scorers or not pass_rule:
        return None
    scorers: list[ExperimentScorerDesignV1] = []
    for raw in raw_scorers:
        dimensions = tuple(
            ExperimentScoreDefinitionV1(
                id=str(item.get("id") or ""),
                label=str(item.get("label") or ""),
                description=str(item.get("description") or "").strip() or None,
                source_key=str(item.get("source_key") or "").strip() or None,
                target=item.get("target"),
                primary=bool(item.get("primary", False)),
            )
            for item in raw.get("dimensions") or ()
            if isinstance(item, Mapping)
        )
        revision = str(raw.get("revision") or "").strip() or None
        links = (
            (
                {
                    "system": "fugue",
                    "kind": "scorer_revision",
                    "ref": revision,
                },
            )
            if revision
            else ()
        )
        scorers.append(
            ExperimentScorerDesignV1(
                id=str(raw.get("id") or ""),
                label=str(raw.get("label") or ""),
                kind=str(raw.get("kind") or ""),  # type: ignore[arg-type]
                description=str(raw.get("description") or ""),
                required=bool(raw.get("required", True)),
                threshold=_optional_float(raw.get("threshold")),
                aggregation=str(raw.get("aggregation") or "").strip() or None,
                evidence_inputs=tuple(
                    str(item)
                    for item in raw.get("evidence_inputs") or ()
                    if str(item).strip()
                ),
                revision=revision,
                model=str(raw.get("model") or "").strip() or None,
                rubric_summary=(
                    str(raw.get("rubric_summary") or "").strip() or None
                ),
                blind_fields=tuple(
                    str(item)
                    for item in raw.get("blind_fields") or ()
                    if str(item).strip()
                ),
                dimensions=dimensions,
                evidence_links=links,
            )
        )
    return ExperimentEvaluationDesignV1(
        pass_rule=pass_rule,
        scorers=tuple(scorers),
        llm_judge_used=any(item.kind == "llm_judge" for item in scorers),
    )


def build_progress_view(
    record: Mapping[str, Any], run_summary: Mapping[str, Any]
) -> ExperimentViewV1:
    preview = _mapping(record.get("preview"), "preview")
    plan = _mapping_or_empty(preview.get("plan_receipt"))
    plan_cells = {
        str(item.get("coordinate_id") or item.get("cell_id") or ""): item
        for item in plan.get("cells") or ()
        if isinstance(item, Mapping)
    }
    cells = [
        _running_cell(item, plan_cells)
        for item in _sequence(run_summary.get("cells"), "run_summary.cells")
    ]
    state_counts = _evaluation_state_counts(cells)
    completed = sum(
        item.execution_status
        in {"completed", "failed", "cancelled", "interrupted", "not_applicable"}
        for item in cells
    )
    displayed = tuple(cells[:EXPERIMENT_VIEW_CELL_LIMIT])
    return experiment_view_from_dict(
        ExperimentViewV1(
            schema_version=EXPERIMENT_VIEW_SCHEMA_VERSION,
            kind="progress",
            matrix_size=int(preview.get("estimated_cells") or len(cells)),
            preview_digest=str(preview.get("preview_digest") or "") or None,
            approval_state="approved"
            if record.get("approval")
            else "awaiting_approval",
            cell_limit=int(preview.get("estimated_cells") or len(cells)),
            reserved_cost_usd=_record_reserved_cost(record),
            phase=_phase(
                str(record.get("state") or ""), str(run_summary.get("status") or "")
            ),
            completed_cells=completed,
            state_counts=state_counts,
            cells=displayed,
            omitted_cells=max(0, len(cells) - len(displayed)),
        ).to_dict()
    )


def build_evaluation_view(record: Mapping[str, Any]) -> ExperimentViewV1:
    preview = _mapping(record.get("preview"), "preview")
    draft = _mapping(preview.get("draft"), "draft")
    research_view = _mapping_or_empty(draft.get("research_view"))
    evaluation_design = _research_evaluation_design(research_view)
    outcome = _mapping_or_empty(record.get("outcome"))
    rows = [item for item in outcome.get("row_refs") or () if isinstance(item, Mapping)]
    evidence_by_prediction = {
        str(item.get("prediction_id") or ""): item
        for item in outcome.get("evidence_refs") or ()
        if isinstance(item, Mapping)
    }
    evaluation = _mapping_or_empty(record.get("evaluation"))
    evaluation_by_prediction = {
        str(item.get("prediction_id") or ""): item
        for item in evaluation.get("prediction_results") or ()
        if isinstance(item, Mapping)
    }
    authored_evaluation_configured = bool(
        evaluation_by_prediction
        or (
            evaluation_design
            and any(
                scorer.kind in {"criteria", "llm_judge"}
                for scorer in evaluation_design.scorers
            )
        )
    )
    cells = [
        _outcome_cell(
            item,
            evidence_by_prediction,
            evaluation_by_prediction,
            evaluation_design=evaluation_design,
            authored_evaluation_configured=authored_evaluation_configured,
        )
        for item in rows
    ]
    displayed = tuple(cells[:EXPERIMENT_VIEW_CELL_LIMIT])
    arm_totals = _arm_totals(rows)
    measures = _behavioral_measures(rows)
    limitations = _public_limitations(outcome)
    run_status = str(outcome.get("run_status") or "")
    infrastructure_health = (
        "healthy"
        if run_status in {"passed", "failed"} and bool(outcome.get("eligible"))
        else "failed"
        if run_status in {"cancelled", "interrupted"}
        else "unavailable"
    )
    return experiment_view_from_dict(
        ExperimentViewV1(
            schema_version=EXPERIMENT_VIEW_SCHEMA_VERSION,
            kind="evaluation",
            matrix_size=int(
                outcome.get("expected_predictions")
                or preview.get("estimated_cells")
                or len(rows)
            ),
            preview_digest=str(preview.get("preview_digest") or "") or None,
            approval_state="approved"
            if record.get("approval")
            else "awaiting_approval",
            cell_limit=int(preview.get("estimated_cells") or len(rows)),
            reserved_cost_usd=_record_reserved_cost(record),
            phase="completed",
            completed_cells=int(outcome.get("observed_predictions") or len(rows)),
            observed_cost_usd=float(outcome.get("observed_cost_usd") or 0.0),
            state_counts=_evaluation_state_counts(cells),
            cells=displayed,
            omitted_cells=max(0, len(cells) - len(displayed)),
            infrastructure_health=infrastructure_health,
            arm_totals=arm_totals,
            aligned_comparisons=_aligned_comparisons(outcome),
            behavioral_measures=measures,
            outcome_summaries=_outcome_summaries(
                cells,
                infrastructure_health=infrastructure_health,
                evidence_eligible=bool(outcome.get("eligible")),
            ),
            score_summaries=_score_summaries(cells),
            evidence_eligible=bool(outcome.get("eligible")),
            limitations=limitations,
            evidence_links=_record_evidence_links(record),
        ).to_dict()
    )


def _planned_cell(
    raw: Mapping[str, Any], varied_names: Sequence[str]
) -> ExperimentCellViewV1:
    cell_id = _opaque_cell_id(raw)
    return ExperimentCellViewV1(
        cell_id=cell_id,
        task_label=_reviewed_task_label(raw),
        factor_levels={
            name: (_value_for_dimension(name, raw) or "fixed") for name in varied_names
        },
        attempt=max(1, int(raw.get("trial_index") or 1)),
        execution_status=(
            "queued" if bool(raw.get("applicable", True)) else "not_applicable"
        ),
        task_outcome=(
            "pending" if bool(raw.get("applicable", True)) else "not_applicable"
        ),
        evaluation_status=(
            "pending" if bool(raw.get("applicable", True)) else "not_applicable"
        ),
        evidence_status=(
            "pending" if bool(raw.get("applicable", True)) else "not_applicable"
        ),
        reason_code=(None if bool(raw.get("applicable", True)) else "not_applicable"),
    )


def _running_cell(
    raw: Mapping[str, Any], plan_cells: Mapping[str, Mapping[str, Any]]
) -> ExperimentCellViewV1:
    status = _execution_status(str(raw.get("status") or "queued"))
    coordinate = str(raw.get("cell_id") or "")
    plan = plan_cells.get(coordinate, {})
    factor_levels = {
        "harness": str(raw.get("harness") or plan.get("harness") or ""),
        "variant": str(raw.get("variant_id") or plan.get("variant_id") or ""),
    }
    factor_levels = {key: value for key, value in factor_levels.items() if value}
    outcome = _benchmark_outcome(raw.get("benchmark_outcome"), status)
    return ExperimentCellViewV1(
        cell_id=_opaque_cell_id({**plan, **raw}),
        task_label=_reviewed_task_label({**plan, **raw}),
        factor_levels=factor_levels,
        attempt=max(1, int(plan.get("trial_index") or 1)),
        execution_status=status,
        task_outcome=outcome,
        evaluation_status=(
            "not_applicable" if status == "not_applicable" else "pending"
        ),
        evidence_status=("not_applicable" if status == "not_applicable" else "pending"),
        reason_code=_safe_reason(status, outcome),
        latency_sec=_optional_float(raw.get("wall_time_sec")),
    )


def _outcome_cell(
    row: Mapping[str, Any],
    evidence_by_prediction: Mapping[str, Mapping[str, Any]],
    evaluation_by_prediction: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_design: ExperimentEvaluationDesignV1 | None,
    authored_evaluation_configured: bool,
) -> ExperimentCellViewV1:
    prediction_id = str(row.get("prediction_id") or "")
    evidence = evidence_by_prediction.get(prediction_id, {})
    evaluation_row = evaluation_by_prediction.get(prediction_id, {})
    execution = _execution_status(str(row.get("status") or "completed"))
    outcome = _row_outcome(row, execution)
    evaluation = _row_evaluation(
        evaluation_row,
        execution,
        configured=authored_evaluation_configured,
    )
    trace_status = str(row.get("trace_link_status") or "")
    evidence_status: EvidenceStatus = (
        "not_applicable"
        if execution == "not_applicable"
        else "reconciled"
        if trace_status in {"ok", "linked", "reconciled"} or evidence
        else "missing"
    )
    links: list[dict[str, str]] = []
    if prediction_id:
        links.append(
            {
                "system": "fugue",
                "kind": "prediction",
                "ref": prediction_id,
            }
        )
    candidate_id = str(row.get("candidate_id") or "")
    if candidate_id:
        links.append(
            {
                "system": "fugue",
                "kind": "route_runtime_receipt",
                "ref": candidate_id,
            }
        )
    run_snapshot = str(row.get("run_snapshot_sha256") or "")
    if run_snapshot:
        links.append(
            {
                "system": "fugue",
                "kind": "run_snapshot",
                "ref": str(row.get("run_id") or prediction_id),
                "digest": run_snapshot,
            }
        )
    source_commit = str(row.get("source_commit") or "")
    if source_commit:
        links.append(
            {
                "system": "git",
                "kind": "source_commit",
                "ref": source_commit,
            }
        )
    trace_project = str(row.get("trace_project") or "")
    prediction_call_id = str(row.get("weave_prediction_call_id") or "")
    if (
        len(trace_project.split("/")) == 2
        and all(trace_project.split("/"))
        and prediction_call_id
    ):
        links.append(
            {
                "system": "weave",
                "kind": "agent_conversation",
                "ref": f"{trace_project}/call/{prediction_call_id}",
            }
        )
    evaluation_call_id = str(row.get("eval_predict_and_score_call_id") or "")
    if (
        len(trace_project.split("/")) == 2
        and all(trace_project.split("/"))
        and evaluation_call_id
    ):
        links.append(
            {
                "system": "weave",
                "kind": "evaluation_attempt",
                "ref": f"{trace_project}/call/{evaluation_call_id}",
            }
        )
    agent_url = str(evidence.get("agent_url") or "")
    if agent_url.startswith("https://"):
        links.append(
            {
                "system": "weave",
                "kind": "agent_conversation",
                "ref": prediction_id,
                "uri": agent_url,
            }
        )
    for conversation_id in (
        evidence.get("conversation_ids") or row.get("weave_conversation_ids") or ()
    ):
        if conversation_id:
            links.append(
                {
                    "system": "weave",
                    "kind": "conversation_identity",
                    "ref": str(conversation_id),
                }
            )
    for root_span_id in (
        evidence.get("root_span_ids") or row.get("weave_root_span_ids") or ()
    ):
        if root_span_id:
            links.append(
                {
                    "system": "weave",
                    "kind": "invoke_agent_root",
                    "ref": str(root_span_id),
                }
            )
    for trace_id in evidence.get("trace_ids") or row.get("weave_trace_ids") or ():
        if trace_id:
            links.append(
                {
                    "system": "weave",
                    "kind": "trace",
                    "ref": str(trace_id),
                }
            )
    run_id = str(row.get("run_id") or "")
    if run_id:
        links.append(
            {
                "system": "fugue",
                "kind": "run",
                "ref": run_id,
            }
        )
    factor_levels = {
        "harness": str(row.get("harness") or ""),
        "variant": str(row.get("variant_id") or ""),
        "context": str(row.get("context_system_id") or ""),
    }
    return ExperimentCellViewV1(
        cell_id=_opaque_cell_id(row),
        task_label=str(
            row.get("task_name") or row.get("comparison_example_id") or "Reviewed task"
        )[:200],
        factor_levels={key: value for key, value in factor_levels.items() if value},
        attempt=max(1, int(row.get("trial_index") or 1)),
        execution_status=execution,
        task_outcome=outcome,
        evaluation_status=evaluation,
        evidence_status=evidence_status,
        reason_code=_safe_reason(execution, outcome, evidence_status),
        cost_usd=_optional_float(
            row.get("cost_usd") or row.get("weave_total_cost_usd")
        ),
        latency_sec=_optional_float(row.get("wall_time_sec")),
        evidence_links=tuple(links),
        measures={
            key: row[key]
            for key in _SAFE_BEHAVIORAL_MEASURES
            if row.get(key) is not None
            and isinstance(row[key], str | int | float | bool)
        },
        scores=_attempt_scores(row, evaluation_row, evaluation_design),
    )


def _levels_for(
    name: str, draft: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    values = [_value_for_dimension(name, item) for item in cells]
    if not any(values):
        normalized = name.lower().replace("-", "_").replace(" ", "_")
        words = set(normalized.replace(",", "_").split("_"))
        aliases = {
            "harness": "harnesses",
            "harnesses": "harnesses",
            "variant": "variants",
            "loop": "variants",
            "loop_design": "variants",
            "context": "context_systems",
            "context_system": "context_systems",
            "workload": "workloads",
        }
        source_key = aliases.get(normalized, "")
        if not source_key and (
            "harness" in words or {"codex", "claude"}.issubset(words)
        ):
            source_key = "harnesses"
        elif not source_key and ("loop" in words or "variant" in words):
            source_key = "variants"
        elif not source_key and "task" in words:
            source_key = "workloads"
        source = draft.get(source_key)
        if source:
            values.extend(str(item) for item in source)
        elif "model" in words and draft.get("model"):
            values.append(str(draft["model"]))
        elif words.intersection({"attempt", "attempts"}):
            values.append(str(draft.get("n_attempts") or 1))
        elif words.intersection({"runtime", "prompt", "tools", "tool"}):
            values.append("held fixed")
        elif words.intersection({"environment", "harbor", "network"}):
            values.append(
                "Harbor · no external network"
                if {"without", "external", "network"}.issubset(words)
                else "Harbor"
            )
    return tuple(_ordered_values([value for value in values if value]))


def _factor_level_labels(
    name: str,
    draft: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
) -> dict[str, str]:
    return {
        level: labels[level]
        for level in _levels_for(name, draft, cells)
        if level in labels
    }


def _display_labels(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    values = _mapping(raw, "display_labels")
    if len(values) > 128:
        raise ValueError("display_labels may contain at most 128 entries")
    return {
        _text(key, "display label id", 300): _text(value, "display label", 300)
        for key, value in values.items()
    }


def _dimension_label(name: str) -> str:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    words = set(normalized.replace(",", "_").split("_"))
    if "harness" in words or {"codex", "claude"}.issubset(words):
        return "harness"
    if "loop" in words or "variant" in words:
        return "variant"
    if "model" in words:
        return "model and sampling"
    if "task" in words:
        return "taskset"
    if words.intersection({"tools", "tool", "runtime", "prompt"}):
        return "tools, runtime, and prompt"
    if words.intersection({"environment", "harbor", "network"}):
        return "environment"
    if words.intersection({"attempt", "attempts"}):
        return "attempt"
    return name


def _value_for_dimension(name: str, cell: Mapping[str, Any]) -> str:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "harness": "harness",
        "harnesses": "harness",
        "variant": "variant_id",
        "loop": "variant_id",
        "loop_design": "variant_id",
        "context": "context_system_id",
        "context_system": "context_system_id",
        "model": "model",
        "workload": "workload_id",
        "task": "task_id",
        "tasks": "task_id",
        "attempt": "trial_index",
        "attempts": "trial_index",
    }
    key = aliases.get(normalized, normalized)
    value = cell.get(key)
    if value is None:
        return ""
    return str(value)


def _arm_totals(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("variant_id") or "default")
        harness = str(row.get("harness") or "all")
        grouped[(variant, "all")].append(row)
        if harness != "all":
            grouped[(variant, harness)].append(row)
    result = []
    for (variant, harness), arm_rows in sorted(grouped.items()):
        passed = sum(1 for row in arm_rows if row.get("pass") is True)
        result.append(
            {
                "arm": variant,
                "harness": harness,
                "passed": passed,
                "total": len(arm_rows),
            }
        )
    return tuple(result)


def _behavioral_measures(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _SAFE_BEHAVIORAL_MEASURES:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            continue
        numeric = [float(value) for value in values if isinstance(value, int | float)]
        if len(numeric) == len(values):
            result[key] = {
                "observed": len(numeric),
                "mean": sum(numeric) / len(numeric),
            }
        else:
            result[key] = {"observed": len(values)}
    return result


def _attempt_scores(
    row: Mapping[str, Any],
    evaluation_row: Mapping[str, Any],
    design: ExperimentEvaluationDesignV1 | None,
) -> tuple[ExperimentScoreResultV1, ...]:
    if design is None:
        return ()
    criteria = {
        str(item.get("criterion_id") or ""): item
        for item in evaluation_row.get("criteria") or ()
        if isinstance(item, Mapping)
    }
    results: list[ExperimentScoreResultV1] = []
    for scorer in design.scorers:
        if not scorer.dimensions:
            value = evaluation_row.get("criteria_score")
            passed = evaluation_row.get("criteria_pass")
            results.append(
                ExperimentScoreResultV1(
                    id=scorer.id,
                    label=scorer.label,
                    status=(
                        "passed"
                        if passed is True
                        else "failed"
                        if passed is False
                        else "unavailable"
                    ),
                    value=value if isinstance(value, str | int | float | bool) else None,
                    scorer_id=scorer.id,
                )
            )
            continue
        for dimension in scorer.dimensions:
            source_key = dimension.source_key or dimension.id
            criterion = criteria.get(source_key, {})
            value = row.get(source_key)
            if value is None and criterion:
                value = criterion.get("score")
            if value is not None and not isinstance(value, str | int | float | bool):
                value = None
            status: ScoreStatus
            if value is None:
                status = "unavailable"
            elif dimension.target is not None:
                status = "passed" if value == dimension.target else "failed"
            elif criterion.get("passed") is True:
                status = "passed"
            elif criterion.get("passed") is False:
                status = "failed"
            else:
                status = "observed"
            results.append(
                ExperimentScoreResultV1(
                    id=dimension.id,
                    label=dimension.label,
                    status=status,
                    value=value,
                    scorer_id=scorer.id,
                )
            )
    return tuple(results)


def _score_summaries(
    cells: Sequence[ExperimentCellViewV1],
) -> tuple[ExperimentScoreSummaryV1, ...]:
    grouped: dict[str, list[ExperimentScoreResultV1]] = defaultdict(list)
    labels: dict[str, str] = {}
    for cell in cells:
        for score in cell.scores:
            grouped[score.id].append(score)
            labels.setdefault(score.id, score.label)
    summaries: list[ExperimentScoreSummaryV1] = []
    for score_id, values in grouped.items():
        numeric = [
            float(item.value)
            for item in values
            if isinstance(item.value, int | float)
        ]
        passed = sum(item.status == "passed" for item in values)
        failed = sum(item.status == "failed" for item in values)
        summaries.append(
            ExperimentScoreSummaryV1(
                id=score_id,
                label=labels[score_id],
                observed=sum(
                    item.status not in {"unavailable", "not_applicable"}
                    for item in values
                ),
                passed=passed if passed or failed else None,
                failed=failed if passed or failed else None,
                unavailable=sum(item.status == "unavailable" for item in values),
                mean=(sum(numeric) / len(numeric) if numeric else None),
            )
        )
    return tuple(summaries)


def _outcome_summaries(
    cells: Sequence[ExperimentCellViewV1],
    *,
    infrastructure_health: str,
    evidence_eligible: bool,
) -> tuple[ExperimentOutcomeSummaryV1, ...]:
    return (
        _cell_outcome_summary(
            cells,
            id="deterministic_task",
            label="Task outcome",
            field_name="task_outcome",
        ),
        _cell_outcome_summary(
            cells,
            id="authored_evaluation",
            label="Authored evaluation",
            field_name="evaluation_status",
        ),
        ExperimentOutcomeSummaryV1(
            id="infrastructure",
            label="Infrastructure",
            status=(
                "passed"
                if infrastructure_health == "healthy"
                else "failed"
                if infrastructure_health == "failed"
                else "unavailable"
            ),
        ),
        ExperimentOutcomeSummaryV1(
            id="evidence",
            label="Evidence",
            status="passed" if evidence_eligible else "failed",
        ),
    )


def _cell_outcome_summary(
    cells: Sequence[ExperimentCellViewV1],
    *,
    id: str,
    label: str,
    field_name: Literal["task_outcome", "evaluation_status"],
) -> ExperimentOutcomeSummaryV1:
    values = [getattr(cell, field_name) for cell in cells]
    scored = [value for value in values if value in {"passed", "failed"}]
    unavailable = sum(value == "unavailable" for value in values)
    not_applicable = sum(value == "not_applicable" for value in values)
    if not scored:
        status: SummaryStatus = (
            "not_applicable"
            if values and not_applicable == len(values)
            else "unavailable"
        )
        return ExperimentOutcomeSummaryV1(
            id=id,
            label=label,
            status=status,
            unavailable=unavailable,
        )
    passed = sum(value == "passed" for value in scored)
    return ExperimentOutcomeSummaryV1(
        id=id,
        label=label,
        status="passed" if passed == len(scored) else "failed",
        passed=passed,
        total=len(scored),
        unavailable=unavailable,
    )


def _aligned_comparisons(outcome: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    values = []
    for item in outcome.get("analysis_results") or ():
        if not isinstance(item, Mapping):
            continue
        analysis_id = str(item.get("analysis_id") or item.get("id") or "")
        digest = str(item.get("analysis_digest") or item.get("digest") or "")
        if analysis_id:
            values.append(
                {
                    "analysis_id": analysis_id[:300],
                    **({"digest": digest} if digest else {}),
                }
            )
    return tuple(values)


def _public_limitations(outcome: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    if outcome.get("limitations"):
        values.append(
            "Additional limitations are recorded in the immutable Fugue outcome."
        )
    if outcome.get("eligibility_failures"):
        values.append(
            "One or more evidence-eligibility requirements were not satisfied."
        )
    unmeasured = int(outcome.get("unmeasured_cost_cells") or 0)
    if unmeasured:
        values.append(f"Observed cost is unavailable for {unmeasured} cells.")
    return tuple(values)


def _evaluation_state_counts(cells: Sequence[ExperimentCellViewV1]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for cell in cells:
        counts[f"execution:{cell.execution_status}"] += 1
        counts[f"task:{cell.task_outcome}"] += 1
        counts[f"evaluation:{cell.evaluation_status}"] += 1
        counts[f"evidence:{cell.evidence_status}"] += 1
    return dict(sorted(counts.items()))


def _row_outcome(row: Mapping[str, Any], execution: ExecutionStatus) -> OutcomeStatus:
    if execution == "not_applicable":
        return "not_applicable"
    if execution in {"failed", "cancelled", "interrupted"}:
        return "unavailable"
    if row.get("pass") is True:
        return "passed"
    if row.get("pass") is False:
        return "failed"
    return "unavailable"


def _row_evaluation(
    row: Mapping[str, Any],
    execution: ExecutionStatus,
    *,
    configured: bool,
) -> OutcomeStatus:
    if execution == "not_applicable":
        return "not_applicable"
    if not configured:
        return "not_applicable"
    for key in ("criteria_pass", "authored_pass", "evaluation_pass"):
        if row.get(key) is True:
            return "passed"
        if row.get(key) is False:
            return "failed"
    return "unavailable"


def _benchmark_outcome(value: Any, execution: ExecutionStatus) -> OutcomeStatus:
    if execution == "not_applicable":
        return "not_applicable"
    normalized = str(value or "").lower()
    if normalized in {"passed", "pass"}:
        return "passed"
    if normalized in {"failed", "fail"}:
        return "failed"
    if execution in {"failed", "cancelled", "interrupted"}:
        return "unavailable"
    return "pending"


def _execution_status(value: str) -> ExecutionStatus:
    normalized = value.lower()
    aliases = {
        "pending": "queued",
        "created": "queued",
        "starting": "preparing",
        "launching": "preparing",
        "passed": "completed",
        "succeeded": "completed",
        "skipped": "not_applicable",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _EXECUTION_STATES:
        return "failed"
    return normalized  # type: ignore[return-value]


def _phase(record_state: str, run_state: str) -> str:
    if run_state in {"running", "starting", "launching"}:
        return "run"
    if record_state in {"scoring", "analyzing"}:
        return "evaluation"
    return "preparation"


def _safe_reason(
    execution: ExecutionStatus,
    outcome: OutcomeStatus,
    evidence: EvidenceStatus = "pending",
) -> str | None:
    if execution in {"failed", "cancelled", "interrupted"}:
        return f"execution_{execution}"
    if evidence == "missing":
        return "evidence_missing"
    if outcome == "failed":
        return "task_not_passed"
    if outcome == "unavailable":
        return "task_outcome_unavailable"
    if execution == "not_applicable":
        return "not_applicable"
    return None


def _opaque_cell_id(raw: Mapping[str, Any]) -> str:
    existing = str(raw.get("coordinate_id") or raw.get("cell_id") or "")
    if existing:
        return existing[:300]
    identity = {
        key: raw.get(key)
        for key in (
            "candidate_id",
            "comparison_example_id",
            "workload_id",
            "task_id",
            "harness",
            "variant_id",
            "context_system_id",
            "trial_index",
        )
        if raw.get(key) is not None
    }
    return f"cell-{stable_digest(identity)[:16]}"


def _reviewed_task_label(raw: Mapping[str, Any]) -> str:
    value = raw.get("task_name") or raw.get("task_label") or raw.get("task_id")
    return str(value or "Reviewed task")[:200]


def _taskset_label(draft: Mapping[str, Any], task_count: Any) -> str:
    workloads = [str(item) for item in draft.get("workloads") or ()]
    if workloads:
        return ", ".join(_humanize(item) for item in workloads)[:300]
    count = int(task_count or 0)
    return f"{count} locked task{'s' if count != 1 else ''}"


def _record_reserved_cost(record: Mapping[str, Any]) -> float | None:
    admission = _mapping_or_empty(record.get("admission"))
    preview = _mapping_or_empty(record.get("preview"))
    value = admission.get("reserved_cost_usd", preview.get("estimated_cost_usd"))
    return _optional_float(value)


def _record_evidence_links(record: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    links: list[dict[str, str]] = []
    preview = _mapping_or_empty(record.get("preview"))
    draft = _mapping_or_empty(preview.get("draft"))
    recipe = _mapping_or_empty(draft.get("task_recipe_preview"))
    provenance = _mapping_or_empty(recipe.get("provenance"))
    public_source = _mapping_or_empty(record.get("public_source_evidence"))
    project = str(public_source.get("project") or provenance.get("project") or "")
    selected_call_ids = (
        public_source.get("selected_call_ids")
        or provenance.get("selected_call_ids")
        or ()
    )
    if len(project.split("/")) == 2 and all(project.split("/")):
        for call_id in selected_call_ids:
            if isinstance(call_id, str) and call_id:
                links.append(
                    {
                        "system": "weave",
                        "kind": "source_call",
                        "ref": f"{project}/call/{call_id}",
                    }
                )
    run_id = str(record.get("run_id") or "")
    if run_id:
        links.append({"system": "fugue", "kind": "run", "ref": run_id})
    outcome = _mapping_or_empty(record.get("outcome"))
    outcome_digest = str(outcome.get("outcome_digest") or "")
    if outcome_digest:
        links.append(
            {
                "system": "fugue",
                "kind": "outcome",
                "ref": str(outcome.get("outcome_id") or run_id or outcome_digest),
                "digest": outcome_digest,
            }
        )
    evaluation = _mapping_or_empty(record.get("evaluation"))
    evaluation_digest = str(evaluation.get("evaluation_digest") or "")
    if evaluation_digest:
        links.append(
            {
                "system": "fugue",
                "kind": "evaluation",
                "ref": str(
                    evaluation.get("evaluation_id")
                    or evaluation.get("scoring_revision_id")
                    or evaluation_digest
                ),
                "digest": evaluation_digest,
            }
        )
    analysis = _mapping_or_empty(record.get("analysis"))
    analysis_digest = str(analysis.get("analysis_digest") or "")
    if analysis_digest:
        links.append(
            {
                "system": "fugue",
                "kind": "analysis",
                "ref": str(analysis.get("analysis_id") or analysis_digest),
                "digest": analysis_digest,
            }
        )
    return _evidence_links(links)


def _validate_view_shape(view: ExperimentViewV1) -> None:
    if view.kind == "design":
        if (
            not view.question
            or not view.hypothesis
            or view.taskset is None
            or view.runtime is None
        ):
            raise ValueError(
                "design view requires question, hypothesis, taskset, and runtime"
            )
        _reject_cross_kind_values(
            view,
            (
                "phase",
                "completed_cells",
                "observed_cost_usd",
                "state_counts",
                "infrastructure_health",
                "arm_totals",
                "aligned_comparisons",
                "behavioral_measures",
                "outcome_summaries",
                "score_summaries",
                "evidence_eligible",
                "limitations",
                "evidence_links",
            ),
        )
    if view.kind == "progress" and not view.phase:
        raise ValueError("progress view requires phase")
    if view.kind == "progress":
        _reject_cross_kind_values(
            view,
            (
                "question",
                "hypothesis",
                "context",
                "observation",
                "rationale",
                "alternative_explanations",
                "success_definition",
                "task_design",
                "prompt_design",
                "evaluation_design",
                "research_label",
                "study_label",
                "source_cohort",
                "fixed_conditions",
                "varied_factors",
                "measured_outcomes",
                "taskset",
                "harnesses",
                "runtime",
                "infrastructure_health",
                "arm_totals",
                "aligned_comparisons",
                "behavioral_measures",
                "outcome_summaries",
                "score_summaries",
                "evidence_eligible",
                "limitations",
                "evidence_links",
            ),
        )
    if view.kind == "evaluation" and view.evidence_eligible is None:
        raise ValueError("evaluation view requires evidence eligibility")
    if view.kind == "evaluation":
        _reject_cross_kind_values(
            view,
            (
                "question",
                "hypothesis",
                "context",
                "observation",
                "rationale",
                "alternative_explanations",
                "success_definition",
                "task_design",
                "prompt_design",
                "evaluation_design",
                "research_label",
                "study_label",
                "source_cohort",
                "fixed_conditions",
                "varied_factors",
                "measured_outcomes",
                "taskset",
                "harnesses",
                "runtime",
            ),
        )
    if view.completed_cells is not None and view.completed_cells > view.matrix_size:
        raise ValueError("completed_cells cannot exceed matrix_size")
    if view.omitted_cells and len(view.cells) + view.omitted_cells > view.matrix_size:
        raise ValueError("displayed and omitted cells cannot exceed matrix_size")


def _reject_cross_kind_values(view: ExperimentViewV1, fields: Sequence[str]) -> None:
    for name in fields:
        value = getattr(view, name)
        if value is not None and value not in ((), {}):
            raise ValueError(f"{view.kind} view cannot contain {name}")


def _optional_task_design(raw: Any) -> ExperimentTaskDesignV1 | None:
    if raw is None:
        return None
    value = _mapping(raw, "task_design")
    _reject_unknown(
        value,
        {
            "title",
            "summary",
            "interaction_mode",
            "tools",
            "resources",
            "evidence_links",
        },
        "task_design",
    )
    return ExperimentTaskDesignV1(
        title=_text(value.get("title"), "task_design.title", 300),
        summary=_text(value.get("summary"), "task_design.summary", 4000),
        interaction_mode=_optional_text(
            value.get("interaction_mode"), "task_design.interaction_mode", 200
        ),
        tools=tuple(
            _text(item, "task_design.tool", 200)
            for item in _sequence(value.get("tools"), "task_design.tools")
        ),
        resources=tuple(
            _text(item, "task_design.resource", 300)
            for item in _sequence(value.get("resources"), "task_design.resources")
        ),
        evidence_links=_evidence_links(value.get("evidence_links")),
    )


def _optional_prompt_design(raw: Any) -> ExperimentPromptDesignV1 | None:
    if raw is None:
        return None
    value = _mapping(raw, "prompt_design")
    _reject_unknown(
        value,
        {"base_instruction_summary", "treatment_summaries", "evidence_links"},
        "prompt_design",
    )
    treatments = {
        _text(key, "prompt_design treatment id", 200): _text(
            item, "prompt_design treatment summary", 2000
        )
        for key, item in _mapping_or_empty(
            value.get("treatment_summaries")
        ).items()
    }
    return ExperimentPromptDesignV1(
        base_instruction_summary=_text(
            value.get("base_instruction_summary"),
            "prompt_design.base_instruction_summary",
            4000,
        ),
        treatment_summaries=treatments,
        evidence_links=_evidence_links(value.get("evidence_links")),
    )


def _score_definition(raw: Any) -> ExperimentScoreDefinitionV1:
    value = _mapping(raw, "score definition")
    _reject_unknown(
        value,
        {"id", "label", "description", "source_key", "target", "primary"},
        "score definition",
    )
    target = value.get("target")
    if target is not None and not isinstance(target, str | int | float | bool):
        raise ValueError("score definition target must be scalar")
    return ExperimentScoreDefinitionV1(
        id=_text(value.get("id"), "score definition id", 200),
        label=_text(value.get("label"), "score definition label", 300),
        description=_optional_text(
            value.get("description"), "score definition description", 1000
        ),
        source_key=_optional_text(
            value.get("source_key"), "score definition source key", 300
        ),
        target=target,
        primary=_optional_bool(value.get("primary"), "score definition primary")
        or False,
    )


def _scorer_design(raw: Any) -> ExperimentScorerDesignV1:
    value = _mapping(raw, "scorer design")
    _reject_unknown(
        value,
        {
            "id",
            "label",
            "kind",
            "description",
            "required",
            "threshold",
            "aggregation",
            "evidence_inputs",
            "revision",
            "model",
            "rubric_summary",
            "blind_fields",
            "dimensions",
            "evidence_links",
        },
        "scorer design",
    )
    kind = _text(value.get("kind"), "scorer design kind", 80)
    if kind not in {"benchmark", "deterministic", "criteria", "llm_judge"}:
        raise ValueError("unknown scorer design kind")
    threshold = _optional_float(value.get("threshold"))
    if threshold is not None and (
        not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("scorer design threshold must be in [0, 1]")
    return ExperimentScorerDesignV1(
        id=_text(value.get("id"), "scorer design id", 200),
        label=_text(value.get("label"), "scorer design label", 300),
        kind=kind,  # type: ignore[arg-type]
        description=_text(
            value.get("description"), "scorer design description", 2000
        ),
        required=_optional_bool(value.get("required"), "scorer design required")
        is not False,
        threshold=threshold,
        aggregation=_optional_text(
            value.get("aggregation"), "scorer design aggregation", 1000
        ),
        evidence_inputs=tuple(
            _text(item, "scorer evidence input", 300)
            for item in _sequence(
                value.get("evidence_inputs"), "scorer design evidence inputs"
            )
        ),
        revision=_optional_text(value.get("revision"), "scorer revision", 300),
        model=_optional_text(value.get("model"), "scorer model", 300),
        rubric_summary=_optional_text(
            value.get("rubric_summary"), "scorer rubric summary", 2000
        ),
        blind_fields=tuple(
            _text(item, "scorer blind field", 200)
            for item in _sequence(value.get("blind_fields"), "scorer blind fields")
        ),
        dimensions=tuple(
            _score_definition(item)
            for item in _sequence(value.get("dimensions"), "scorer dimensions")
        ),
        evidence_links=_evidence_links(value.get("evidence_links")),
    )


def _optional_evaluation_design(
    raw: Any,
) -> ExperimentEvaluationDesignV1 | None:
    if raw is None:
        return None
    value = _mapping(raw, "evaluation_design")
    _reject_unknown(
        value,
        {"pass_rule", "scorers", "llm_judge_used"},
        "evaluation_design",
    )
    scorers = tuple(
        _scorer_design(item)
        for item in _sequence(value.get("scorers"), "evaluation_design.scorers")
    )
    if not scorers:
        raise ValueError("evaluation_design requires at least one scorer")
    judge_used = (
        _optional_bool(value.get("llm_judge_used"), "evaluation_design.llm_judge_used")
        or False
    )
    if judge_used != any(item.kind == "llm_judge" for item in scorers):
        raise ValueError("evaluation_design judge usage disagrees with its scorers")
    return ExperimentEvaluationDesignV1(
        pass_rule=_text(value.get("pass_rule"), "evaluation_design.pass_rule", 4000),
        scorers=scorers,
        llm_judge_used=judge_used,
    )


def _factor(raw: Any, field_name: str) -> ExperimentFactorV1:
    value = _mapping(raw, field_name)
    _reject_unknown(value, {"name", "levels", "label", "level_labels"}, field_name)
    levels = tuple(
        _text(item, f"{field_name}.level", 300)
        for item in _sequence(value.get("levels"), f"{field_name}.levels")
    )
    level_labels = _display_labels(value.get("level_labels"))
    if set(level_labels) - set(levels):
        raise ValueError(f"{field_name}.level_labels names an unknown level")
    return ExperimentFactorV1(
        name=_text(value.get("name"), f"{field_name}.name", 200),
        levels=levels,
        label=_optional_text(value.get("label"), f"{field_name}.label", 300),
        level_labels=level_labels,
    )


def _outcome_summary(raw: Any) -> ExperimentOutcomeSummaryV1:
    value = _mapping(raw, "outcome_summary")
    _reject_unknown(
        value,
        {"id", "label", "status", "passed", "total", "unavailable"},
        "outcome_summary",
    )
    status = _text(value.get("status"), "outcome_summary.status", 80)
    if status not in {"passed", "failed", "unavailable", "not_applicable"}:
        raise ValueError("unknown outcome summary status")
    passed = _optional_non_negative_int(value.get("passed"), "outcome_summary.passed")
    total = _optional_non_negative_int(value.get("total"), "outcome_summary.total")
    unavailable = _non_negative_int(
        value.get("unavailable", 0), "outcome_summary.unavailable"
    )
    if passed is not None and total is None:
        raise ValueError("outcome summary passed count requires total")
    if passed is not None and total is not None and passed > total:
        raise ValueError("outcome summary passed count cannot exceed total")
    return ExperimentOutcomeSummaryV1(
        id=_text(value.get("id"), "outcome_summary.id", 200),
        label=_text(value.get("label"), "outcome_summary.label", 300),
        status=status,  # type: ignore[arg-type]
        passed=passed,
        total=total,
        unavailable=unavailable,
    )


def _score_result(raw: Any) -> ExperimentScoreResultV1:
    value = _mapping(raw, "score result")
    _reject_unknown(
        value,
        {"id", "label", "status", "value", "scorer_id"},
        "score result",
    )
    status = _text(value.get("status"), "score result status", 80)
    if status not in {
        "passed",
        "failed",
        "observed",
        "unavailable",
        "not_applicable",
    }:
        raise ValueError("unknown score result status")
    score_value = value.get("value")
    if score_value is not None and not isinstance(
        score_value, str | int | float | bool
    ):
        raise ValueError("score result value must be scalar")
    return ExperimentScoreResultV1(
        id=_text(value.get("id"), "score result id", 200),
        label=_text(value.get("label"), "score result label", 300),
        status=status,  # type: ignore[arg-type]
        value=score_value,
        scorer_id=_optional_text(value.get("scorer_id"), "score result scorer", 200),
    )


def _score_summary(raw: Any) -> ExperimentScoreSummaryV1:
    value = _mapping(raw, "score summary")
    _reject_unknown(
        value,
        {"id", "label", "observed", "passed", "failed", "unavailable", "mean"},
        "score summary",
    )
    observed = _non_negative_int(value.get("observed", 0), "score summary observed")
    passed = (
        None
        if value.get("passed") is None
        else _non_negative_int(value["passed"], "score summary passed")
    )
    failed = (
        None
        if value.get("failed") is None
        else _non_negative_int(value["failed"], "score summary failed")
    )
    unavailable = _non_negative_int(
        value.get("unavailable", 0), "score summary unavailable"
    )
    if (passed or 0) + (failed or 0) > observed:
        raise ValueError("score summary statuses exceed observed values")
    return ExperimentScoreSummaryV1(
        id=_text(value.get("id"), "score summary id", 200),
        label=_text(value.get("label"), "score summary label", 300),
        observed=observed,
        passed=passed,
        failed=failed,
        unavailable=unavailable,
        mean=_optional_float(value.get("mean")),
    )


def _descriptor(raw: Any, field_name: str) -> ExperimentDescriptorV1:
    value = _mapping(raw, field_name)
    _reject_unknown(value, {"id", "label", "digest", "details"}, field_name)
    details_raw = _mapping_or_empty(value.get("details"))
    details: dict[str, str | int | bool] = {}
    for key, item in details_raw.items():
        if not isinstance(item, str | int | bool):
            raise ValueError(f"{field_name}.details values must be scalar")
        details[_text(key, f"{field_name}.details key", 100)] = item
    return ExperimentDescriptorV1(
        id=_text(value.get("id"), f"{field_name}.id", 1000),
        label=_text(value.get("label"), f"{field_name}.label", 300),
        digest=_optional_digest(value.get("digest"), f"{field_name}.digest"),
        details=details,
    )


def _optional_descriptor(raw: Any, field_name: str) -> ExperimentDescriptorV1 | None:
    return None if raw is None else _descriptor(raw, field_name)


def _cell(raw: Any) -> ExperimentCellViewV1:
    value = _mapping(raw, "cell")
    _reject_unknown(
        value,
        {
            "cell_id",
            "task_label",
            "factor_levels",
            "attempt",
            "execution_status",
            "task_outcome",
            "evaluation_status",
            "evidence_status",
            "reason_code",
            "cost_usd",
            "latency_sec",
            "evidence_links",
            "measures",
            "scores",
        },
        "cell",
    )
    execution = _text(value.get("execution_status"), "execution_status", 80)
    outcome = _text(value.get("task_outcome"), "task_outcome", 80)
    evaluation = _text(value.get("evaluation_status"), "evaluation_status", 80)
    evidence = _text(value.get("evidence_status"), "evidence_status", 80)
    if execution not in _EXECUTION_STATES:
        raise ValueError("unknown execution status")
    if outcome not in _OUTCOME_STATES or evaluation not in _OUTCOME_STATES:
        raise ValueError("unknown outcome status")
    if evidence not in _EVIDENCE_STATES:
        raise ValueError("unknown evidence status")
    factor_levels = {
        _text(key, "factor name", 200): _text(item, "factor level", 300)
        for key, item in _mapping(value.get("factor_levels"), "factor_levels").items()
    }
    measures: dict[str, str | int | float | bool | None] = {}
    for key, item in _mapping_or_empty(value.get("measures")).items():
        if key not in _SAFE_BEHAVIORAL_MEASURES:
            raise ValueError(f"cell contains an unsupported measure: {key}")
        if item is not None and not isinstance(item, str | int | float | bool):
            raise ValueError("cell measure values must be scalar")
        measures[key] = item
    links = _evidence_links(value.get("evidence_links"))
    return ExperimentCellViewV1(
        cell_id=_text(value.get("cell_id"), "cell_id", 300),
        task_label=_text(value.get("task_label"), "task_label", 200),
        factor_levels=factor_levels,
        attempt=_positive_int(value.get("attempt"), "attempt"),
        execution_status=execution,  # type: ignore[arg-type]
        task_outcome=outcome,  # type: ignore[arg-type]
        evaluation_status=evaluation,  # type: ignore[arg-type]
        evidence_status=evidence,  # type: ignore[arg-type]
        reason_code=_optional_text(value.get("reason_code"), "reason_code", 100),
        cost_usd=_optional_cost(value.get("cost_usd"), "cost_usd"),
        latency_sec=_optional_cost(value.get("latency_sec"), "latency_sec"),
        evidence_links=links,
        measures=measures,
        scores=tuple(
            _score_result(item)
            for item in _sequence(value.get("scores"), "cell scores")
        ),
    )


def _evidence_links(raw: Any) -> tuple[dict[str, str], ...]:
    links: list[dict[str, str]] = []
    for item in _sequence(raw, "evidence_links"):
        link = _mapping(item, "evidence_link")
        _reject_unknown(
            link, {"system", "kind", "ref", "uri", "digest"}, "evidence_link"
        )
        projected = {
            "system": _text(link.get("system"), "evidence_link.system", 100),
            "kind": _text(link.get("kind"), "evidence_link.kind", 100),
            "ref": _text(link.get("ref"), "evidence_link.ref", 1000),
        }
        uri = _optional_text(link.get("uri"), "evidence_link.uri", 2000)
        if uri:
            if not uri.startswith("https://"):
                raise ValueError("evidence link URIs must use https")
            projected["uri"] = uri
        digest = _optional_digest(link.get("digest"), "evidence_link.digest")
        if digest:
            projected["digest"] = digest
        links.append(projected)
    return tuple(links)


def _arm_total(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "arm_total")
    _reject_unknown(value, {"arm", "harness", "passed", "total"}, "arm_total")
    return {
        "arm": _text(value.get("arm"), "arm_total.arm", 300),
        "harness": _text(value.get("harness"), "arm_total.harness", 300),
        "passed": _non_negative_int(value.get("passed"), "arm_total.passed"),
        "total": _non_negative_int(value.get("total"), "arm_total.total"),
    }


def _comparison(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "aligned_comparison")
    _reject_unknown(value, {"analysis_id", "digest"}, "aligned_comparison")
    result = {
        "analysis_id": _text(
            value.get("analysis_id"), "aligned_comparison.analysis_id", 300
        )
    }
    digest = _optional_digest(value.get("digest"), "aligned_comparison.digest")
    if digest:
        result["digest"] = digest
    return result


def _measure_mapping(raw: Any, field_name: str) -> dict[str, Any]:
    value = _mapping_or_empty(raw)
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _SAFE_BEHAVIORAL_MEASURES:
            raise ValueError(f"{field_name} contains an unsupported measure: {key}")
        measure = _mapping(item, f"{field_name}.{key}")
        _reject_unknown(measure, {"observed", "mean"}, f"{field_name}.{key}")
        observed = _non_negative_int(
            measure.get("observed"), f"{field_name}.{key}.observed"
        )
        result[key] = {"observed": observed}
        if measure.get("mean") is not None:
            result[key]["mean"] = float(measure["mean"])
    return result


def _count_mapping(raw: Any, field_name: str) -> dict[str, int]:
    return {
        _text(key, f"{field_name} key", 100): _non_negative_int(
            item, f"{field_name}.{key}"
        )
        for key, item in _mapping_or_empty(raw).items()
    }


def _mapping(raw: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return raw


def _mapping_or_empty(raw: Any) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("expected an object")
    return raw


def _sequence(raw: Any, field_name: str) -> Sequence[Any]:
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple):
        raise ValueError(f"{field_name} must be an array")
    return raw


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"{field_name} has unknown fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )


def _text(raw: Any, field_name: str, maximum: int) -> str:
    if not isinstance(raw, str) or not raw.strip() or len(raw) > maximum:
        raise ValueError(f"{field_name} must contain 1 to {maximum} characters")
    return raw.strip()


def _optional_text(raw: Any, field_name: str, maximum: int) -> str | None:
    if raw is None:
        return None
    return _text(raw, field_name, maximum)


def _positive_int(raw: Any, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return raw


def _non_negative_int(raw: Any, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return raw


def _optional_non_negative_int(raw: Any, field_name: str) -> int | None:
    return None if raw is None else _non_negative_int(raw, field_name)


def _optional_bool(raw: Any, field_name: str) -> bool | None:
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return raw


def _optional_cost(raw: Any, field_name: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float) or raw < 0:
        raise ValueError(f"{field_name} must be a non-negative number")
    return float(raw)


def _optional_float(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw)


def _optional_digest(raw: Any, field_name: str) -> str | None:
    if raw is None or raw == "":
        return None
    return _text(raw, field_name, 1000)


def _ordered_values(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def _drop_empty(
    value: Mapping[str, Any], *, preserve_false: bool = False
) -> dict[str, Any]:
    empty = (None, "", (), [], {})
    return {
        key: item
        for key, item in value.items()
        if item not in empty and (preserve_false or item is not False)
    }
