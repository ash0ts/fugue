from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from fugue.bench.context_contracts import (
    ContextCapability,
    ContextDelivery,
    WorkloadRunner,
    context_capabilities,
)
from fugue.bench.files import as_list as _list
from fugue.bench.files import as_mapping as _dict
from fugue.bench.files import require_unique

CONFIG_ROOT = Path("configs") / "fugue"
PROMPTS_DIR = "prompts"
SKILLS_DIR = "skills"
EXPERIMENTS_DIR = "experiments"
AGENT_PRESETS_DIR = "agent-presets"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class LibraryItem:
    id: str
    title: str
    path: str
    sha256: str


@dataclass(frozen=True)
class Prompt:
    id: str
    title: str
    body: str
    path: str
    sha256: str


@dataclass(frozen=True)
class Skill:
    id: str
    title: str
    body: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ContextSelection:
    system_id: str = "none"
    delivery: ContextDelivery = "portable"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegrationSelection:
    id: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentCandidateSpec:
    harness: str
    model: str
    prompt_id: str | None = None
    skills: list[str] = field(default_factory=list)
    context: ContextSelection = field(default_factory=ContextSelection)
    integrations: list[IntegrationSelection] = field(default_factory=list)
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_env: dict[str, str] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    verifier: dict[str, Any] = field(default_factory=dict)
    retry: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class AgentPresetEvidence:
    suite_id: str = ""
    suite_digest: str = ""
    base_commit: str = ""
    run_ids: list[str] = field(default_factory=list)
    analysis_snapshot: str = ""
    metrics: dict[str, float | int | None] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentPreset:
    id: str
    title: str
    role: Literal["maintainer", "operator"]
    base_experiment_id: str
    candidate: AgentCandidateSpec
    evidence: AgentPresetEvidence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def harness(self) -> str:
        return self.candidate.harness

    @property
    def model(self) -> str:
        return self.candidate.model

    @property
    def prompt_id(self) -> str | None:
        return self.candidate.prompt_id

    @property
    def skill_ids(self) -> list[str]:
        return list(self.candidate.skills)

    @property
    def context(self) -> ContextSelection:
        return self.candidate.context

    @property
    def integrations(self) -> list[IntegrationSelection]:
        return list(self.candidate.integrations)

    @property
    def agent_kwargs(self) -> dict[str, Any]:
        return dict(self.candidate.agent_kwargs)

    @property
    def agent_env(self) -> dict[str, str]:
        return dict(self.candidate.agent_env)

    @property
    def environment(self) -> dict[str, Any]:
        return dict(self.candidate.environment)

    @property
    def verifier(self) -> dict[str, Any]:
        return dict(self.candidate.verifier)

    @property
    def retry(self) -> dict[str, Any]:
        return dict(self.candidate.retry)

    @property
    def artifacts(self) -> list[Any]:
        return list(self.candidate.artifacts)

    @property
    def suite_id(self) -> str:
        return self.evidence.suite_id

    @property
    def suite_digest(self) -> str:
        return self.evidence.suite_digest

    @property
    def base_commit(self) -> str:
        return self.evidence.base_commit

    @property
    def run_ids(self) -> list[str]:
        return list(self.evidence.run_ids)

    @property
    def analysis_snapshot(self) -> str:
        return self.evidence.analysis_snapshot

    @property
    def metrics(self) -> dict[str, float | int | None]:
        return dict(self.evidence.metrics)


@dataclass(frozen=True)
class FeatureVariant:
    id: str
    label: str
    prompt_id: str | None = None
    skills: list[str] = field(default_factory=list)
    context: ContextSelection = field(default_factory=ContextSelection)
    integrations: list[IntegrationSelection] = field(default_factory=list)
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_env: dict[str, str] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    verifier: dict[str, Any] = field(default_factory=dict)
    retry: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Any] = field(default_factory=list)
    enabled: bool = True

    @property
    def skill_ids(self) -> list[str]:
        """Internal/export metadata spelling; experiment input is strictly `skills`."""
        return list(self.skills)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BuiltinScorerSelection:
    type: Literal["builtin"]
    id: Literal["harbor-outcome"]


@dataclass(frozen=True)
class RubricScorerSelection:
    type: Literal["rubric"]
    path: str


ScorerSelection = BuiltinScorerSelection | RubricScorerSelection


def scorer_reference(scorer: ScorerSelection) -> str:
    if isinstance(scorer, BuiltinScorerSelection):
        return f"builtin:{scorer.id}"
    return scorer.path


@dataclass(frozen=True)
class WorkloadSpec:
    id: str
    runner: WorkloadRunner
    manifest: Path | None = None
    dataset: str | None = None
    systems: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    harness_assignment: Literal["cross", "latin_square"] = "cross"
    required_capabilities: list[ContextCapability] = field(default_factory=list)
    n_tasks: int | None = None
    n_attempts: int | None = None
    artifacts: list[Any] = field(default_factory=list)
    scorers: list[ScorerSelection] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationSourceSpec:
    kind: Literal["seed", "file", "mcp"]
    path: str | None = None
    text: str | None = None
    server: str | None = None
    tools: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationGenerationSpec:
    suite_id: str
    workload_id: str
    size: int = 8
    sources: list[EvaluationSourceSpec] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchScoreDimensionSpec:
    id: str
    label: str
    description: str = ""
    source_key: str | None = None
    target: str | float | int | bool | None = None
    primary: bool = False


@dataclass(frozen=True)
class ResearchScorerSpec:
    id: str
    label: str
    kind: Literal["benchmark", "deterministic", "criteria", "llm_judge"]
    description: str
    required: bool = True
    threshold: float | None = None
    aggregation: str = ""
    evidence_inputs: list[str] = field(default_factory=list)
    revision: str | None = None
    model: str | None = None
    rubric_summary: str = ""
    blind_fields: list[str] = field(default_factory=list)
    dimensions: list[ResearchScoreDimensionSpec] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentResearchViewSpec:
    observation: str = ""
    rationale: str = ""
    alternative_explanations: list[str] = field(default_factory=list)
    success_definition: str = ""
    task_title: str = ""
    task_summary: str = ""
    interaction_mode: str = ""
    tools: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    base_instruction_summary: str = ""
    treatment_summaries: dict[str, str] = field(default_factory=dict)
    pass_rule: str = ""
    scorers: list[ResearchScorerSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _paths_to_strings(asdict(self))


@dataclass(frozen=True)
class PresetSpec:
    id: str
    workloads: list[str] = field(default_factory=list)
    systems: list[str] = field(default_factory=list)
    harnesses: list[str] = field(default_factory=list)
    n_tasks: int | None = None
    n_attempts: int | None = None
    n_concurrent: int | None = None
    scheduling_seed: str | None = None
    selection_lock_required: bool = False
    workload_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    title: str
    description: str = ""
    manifest: Path = Path("datasets/pilot.yaml")
    model: str | None = None
    builder_model: str | None = None
    judge_model: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    harnesses: list[str] = field(default_factory=list)
    variants: list[FeatureVariant] = field(default_factory=list)
    n_attempts: int | None = None
    n_concurrent: int | None = None
    n_tasks: int | None = None
    jobs_dir: Path | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Any] = field(default_factory=list)
    verifier: dict[str, Any] = field(default_factory=dict)
    retry: dict[str, Any] = field(default_factory=dict)
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_env: dict[str, str] = field(default_factory=dict)
    integrations: list[IntegrationSelection] = field(default_factory=list)
    workloads: list[WorkloadSpec] = field(default_factory=list)
    evaluation_generation: EvaluationGenerationSpec | None = None
    research_view: ExperimentResearchViewSpec | None = None
    presets: list[PresetSpec] = field(default_factory=list)
    default_preset: str | None = None
    trace_content: str = "full"
    debug: bool = False
    quiet: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = _paths_to_strings(asdict(self))
        value["variants"] = [
            _paths_to_strings(variant.to_dict()) for variant in self.variants
        ]
        return value


def library_root(repo_root: Path | None = None) -> Path:
    root = repo_root or Path.cwd()
    return root / CONFIG_ROOT


def validate_id(value: str, *, kind: str = "id") -> str:
    item_id = str(value or "").strip()
    if not _ID_RE.match(item_id):
        raise ValueError(
            f"invalid {kind} {value!r}; use letters, numbers, '.', '_' or '-'"
        )
    return item_id


def list_prompts(repo_root: Path | None = None) -> list[LibraryItem]:
    return _list_markdown_items(library_root(repo_root) / PROMPTS_DIR)


def get_prompt(item_id: str, repo_root: Path | None = None) -> Prompt:
    item_id = validate_id(item_id, kind="prompt id")
    path = library_root(repo_root) / PROMPTS_DIR / f"{item_id}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {item_id}")
    body = path.read_text()
    return Prompt(
        id=item_id,
        title=_title_from_markdown(body) or item_id,
        body=body,
        path=path.as_posix(),
        sha256=_file_sha256(path),
    )


def list_skills(repo_root: Path | None = None) -> list[LibraryItem]:
    root = library_root(repo_root) / SKILLS_DIR
    if not root.exists():
        return []
    items: list[LibraryItem] = []
    for skill_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        skill_path = skill_dir / "SKILL.md"
        if skill_path.is_file():
            items.append(
                _library_item(
                    skill_dir.name,
                    skill_path,
                    _title_from_markdown(skill_path.read_text()),
                )
            )
    return items


def get_skill(item_id: str, repo_root: Path | None = None) -> Skill:
    item_id = validate_id(item_id, kind="skill id")
    path = library_root(repo_root) / SKILLS_DIR / item_id / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"skill not found: {item_id}")
    body = path.read_text()
    return Skill(
        id=item_id,
        title=_title_from_markdown(body) or item_id,
        body=body,
        path=path.as_posix(),
        sha256=_file_sha256(path),
    )


def list_experiments(repo_root: Path | None = None) -> list[LibraryItem]:
    return _list_yaml_items(library_root(repo_root) / EXPERIMENTS_DIR)


def list_agent_presets(repo_root: Path | None = None) -> list[LibraryItem]:
    return _list_yaml_items(library_root(repo_root) / AGENT_PRESETS_DIR)


def get_agent_preset(item_id: str, repo_root: Path | None = None) -> AgentPreset:
    item_id = validate_id(item_id, kind="agent preset id")
    path = library_root(repo_root) / AGENT_PRESETS_DIR / f"{item_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"agent preset not found: {item_id}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("agent preset YAML must be a mapping")
    _reject_unknown(raw, AgentPreset, kind="agent preset")
    preset_id = validate_id(raw.get("id") or item_id, kind="agent preset id")
    if preset_id != item_id:
        raise ValueError(
            f"agent preset file {path.name!r} declares mismatched id {preset_id!r}"
        )
    role = str(raw.get("role") or "")
    if role not in {"maintainer", "operator"}:
        raise ValueError("agent preset role must be maintainer or operator")
    candidate_raw = raw.get("candidate") or {}
    evidence_raw = raw.get("evidence") or {}
    if not isinstance(candidate_raw, dict):
        raise ValueError("agent preset candidate must be a mapping")
    if not isinstance(evidence_raw, dict):
        raise ValueError("agent preset evidence must be a mapping")
    _reject_unknown(candidate_raw, AgentCandidateSpec, kind="agent preset candidate")
    _reject_unknown(evidence_raw, AgentPresetEvidence, kind="agent preset evidence")
    harness = str(candidate_raw.get("harness") or "")
    if harness not in {"hermes", "openclaw", "claude-code", "codex"}:
        raise ValueError(f"unknown agent preset harness: {harness or '<empty>'}")
    model = str(candidate_raw.get("model") or "").strip()
    if not model:
        raise ValueError("agent preset model is required")
    prompt_id = _optional_str(candidate_raw.get("prompt_id"))
    if prompt_id:
        validate_id(prompt_id, kind="prompt id")
        get_prompt(prompt_id, repo_root)
    skill_ids = _string_list(candidate_raw.get("skills"))
    require_unique(skill_ids, kind=f"agent preset {preset_id} skill")
    for skill_id in skill_ids:
        get_skill(skill_id, repo_root)
    suite_digest = str(evidence_raw.get("suite_digest") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", suite_digest):
        raise ValueError("agent preset suite_digest must be a SHA-256 digest")
    base_commit = str(evidence_raw.get("base_commit") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise ValueError("agent preset base_commit must be a full Git commit")
    metrics = _dict(evidence_raw.get("metrics"))
    if any(
        value is not None and not isinstance(value, (int, float))
        for value in metrics.values()
    ):
        raise ValueError("agent preset metrics must be numeric or null")
    return AgentPreset(
        id=preset_id,
        title=str(raw.get("title") or preset_id),
        role=role,  # type: ignore[arg-type]
        base_experiment_id=validate_id(
            raw.get("base_experiment_id") or "pilot", kind="experiment id"
        ),
        candidate=AgentCandidateSpec(
            harness=harness,
            model=model,
            prompt_id=prompt_id,
            skills=skill_ids,
            context=_context_selection(candidate_raw.get("context")),
            integrations=_integration_selections(
                candidate_raw.get("integrations"), kind="agent preset candidate"
            ),
            agent_kwargs=_dict(candidate_raw.get("agent_kwargs")),
            agent_env={
                str(k): str(v) for k, v in _dict(candidate_raw.get("agent_env")).items()
            },
            environment=_dict(candidate_raw.get("environment")),
            verifier=_dict(candidate_raw.get("verifier")),
            retry=_dict(candidate_raw.get("retry")),
            artifacts=_list(candidate_raw.get("artifacts")),
        ),
        evidence=AgentPresetEvidence(
            suite_id=str(evidence_raw.get("suite_id") or ""),
            suite_digest=suite_digest,
            base_commit=base_commit,
            run_ids=_string_list(evidence_raw.get("run_ids")),
            analysis_snapshot=str(evidence_raw.get("analysis_snapshot") or ""),
            metrics={str(key): value for key, value in metrics.items()},
        ),
    )


def get_experiment(item_id: str, repo_root: Path | None = None) -> ExperimentSpec:
    item_id = validate_id(item_id, kind="experiment id")
    path = library_root(repo_root) / EXPERIMENTS_DIR / f"{item_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"experiment not found: {item_id}")
    experiment = experiment_from_yaml(path.read_text(), item_id=item_id)
    if experiment.id != item_id:
        raise ValueError(
            f"experiment file {path.name!r} declares mismatched id {experiment.id!r}"
        )
    return experiment


def save_experiment(
    item_id: str, body: str, repo_root: Path | None = None
) -> ExperimentSpec:
    item_id = validate_id(item_id, kind="experiment id")
    experiment = experiment_from_yaml(body, item_id=item_id)
    if experiment.id != item_id:
        raise ValueError(
            f"experiment body id {experiment.id!r} does not match {item_id!r}"
        )
    path = library_root(repo_root) / EXPERIMENTS_DIR / f"{item_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_normalize_text(body))
    return get_experiment(item_id, repo_root)


def save_experiment_data(
    item_id: str, data: dict[str, Any], repo_root: Path | None = None
) -> ExperimentSpec:
    data = dict(data)
    data["id"] = item_id
    experiment = experiment_from_data(data, item_id=item_id)
    return save_experiment(item_id, experiment_to_yaml(experiment), repo_root)


def experiment_from_yaml(text: str, *, item_id: str | None = None) -> ExperimentSpec:
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError("experiment YAML must be a mapping")
    return experiment_from_data(raw, item_id=item_id)


def experiment_from_data(
    raw: dict[str, Any], *, item_id: str | None = None
) -> ExperimentSpec:
    _reject_unknown(raw, ExperimentSpec, kind="experiment")
    experiment_id = validate_id(
        raw.get("id") or item_id or "experiment", kind="experiment id"
    )
    variants = _variants(raw)
    if not variants:
        variants = [FeatureVariant(id="baseline", label="Baseline")]
    workloads = _workloads(raw.get("workloads"))
    presets = _presets(raw.get("presets"))
    require_unique([variant.id for variant in variants], kind="variant")
    require_unique([workload.id for workload in workloads], kind="workload")
    require_unique([preset.id for preset in presets], kind="preset")
    workload_ids = {workload.id for workload in workloads}
    for preset in presets:
        unknown = sorted(
            (set(preset.workloads) | set(preset.workload_overrides)) - workload_ids
        )
        if unknown:
            raise ValueError(
                f"preset {preset.id} overrides unknown workload(s): {', '.join(unknown)}"
            )
    variant_ids = {variant.id for variant in variants}
    for workload in workloads:
        unknown_variants = sorted(set(workload.variants) - variant_ids)
        if unknown_variants:
            raise ValueError(
                f"workload {workload.id} selects unknown variant(s): "
                + ", ".join(unknown_variants)
            )
    default_preset = _optional_str(raw.get("default_preset"))
    if default_preset and default_preset not in {preset.id for preset in presets}:
        raise ValueError(f"unknown default preset: {default_preset}")
    return ExperimentSpec(
        id=experiment_id,
        title=str(raw.get("title") or experiment_id),
        description=str(raw.get("description") or ""),
        manifest=Path(str(raw.get("manifest") or "datasets/pilot.yaml")),
        model=_optional_str(raw.get("model")),
        builder_model=_optional_str(raw.get("builder_model")),
        judge_model=_optional_str(raw.get("judge_model")),
        run_name=_optional_str(raw.get("run_name")),
        tags=_string_list(raw.get("tags")),
        harnesses=_string_list(raw.get("harnesses")),
        variants=variants,
        n_attempts=_positive_int(raw.get("n_attempts"), kind="experiment n_attempts"),
        n_concurrent=_positive_int(
            raw.get("n_concurrent"), kind="experiment n_concurrent"
        ),
        n_tasks=_positive_int(raw.get("n_tasks"), kind="experiment n_tasks"),
        jobs_dir=Path(str(raw["jobs_dir"])) if raw.get("jobs_dir") else None,
        environment=_dict(raw.get("environment")),
        artifacts=_list(raw.get("artifacts")),
        verifier=_dict(raw.get("verifier")),
        retry=_dict(raw.get("retry")),
        agent_kwargs=_dict(raw.get("agent_kwargs")),
        agent_env={str(k): str(v) for k, v in _dict(raw.get("agent_env")).items()},
        integrations=_integration_selections(
            raw.get("integrations"), kind="experiment"
        ),
        workloads=workloads,
        evaluation_generation=_evaluation_generation(raw.get("evaluation_generation")),
        research_view=research_view_from_data(raw.get("research_view")),
        presets=presets,
        default_preset=default_preset,
        trace_content=_trace_content(raw.get("trace_content")),
        debug=bool(raw.get("debug", False)),
        quiet=bool(raw.get("quiet", False)),
    )


def experiment_to_yaml(experiment: ExperimentSpec) -> str:
    return yaml.safe_dump(experiment.to_dict(), sort_keys=False)


def experiment_with_overrides(
    experiment: ExperimentSpec, **overrides: Any
) -> ExperimentSpec:
    data = experiment.to_dict()
    for key, value in overrides.items():
        if value in (None, "", []):
            continue
        data[key] = value
    return experiment_from_yaml(yaml.safe_dump(data), item_id=experiment.id)


def _variants(raw: dict[str, Any]) -> list[FeatureVariant]:
    values = raw.get("variants")
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("variants must be a list")
    return [_feature_variant(item, index) for index, item in enumerate(values, start=1)]


def _feature_variant(raw: Any, index: int) -> FeatureVariant:
    if not isinstance(raw, dict):
        raise ValueError("variant must be a mapping")
    _reject_unknown(raw, FeatureVariant, kind="variant")
    variant_id = validate_id(raw.get("id") or f"variant-{index}", kind="variant id")
    prompt_id = _optional_str(raw.get("prompt_id"))
    if prompt_id:
        validate_id(prompt_id, kind="prompt id")
    if raw.get("skill_ids") is not None:
        raise ValueError("unknown variant field(s): skill_ids; use skills")
    skills = _string_list(raw.get("skills"))
    for skill_id in skills:
        validate_id(skill_id, kind="skill id")
    require_unique(skills, kind=f"variant {variant_id} skill")
    return FeatureVariant(
        id=variant_id,
        label=str(raw.get("label") or variant_id),
        prompt_id=prompt_id,
        skills=skills,
        context=_context_selection(raw.get("context")),
        integrations=_integration_selections(
            raw.get("integrations"), kind=f"variant {variant_id}"
        ),
        agent_kwargs=_dict(raw.get("agent_kwargs")),
        agent_env={str(k): str(v) for k, v in _dict(raw.get("agent_env")).items()},
        environment=_dict(raw.get("environment")),
        verifier=_dict(raw.get("verifier")),
        retry=_dict(raw.get("retry")),
        artifacts=_list(raw.get("artifacts")),
        enabled=bool(raw.get("enabled", True)),
    )


def _context_selection(raw: Any) -> ContextSelection:
    if raw in (None, ""):
        raise ValueError(
            "variant context must be a mapping with explicit delivery and system_id"
        )
    if isinstance(raw, str):
        raise ValueError(
            "variant context must be a mapping with explicit delivery and system_id"
        )
    if not isinstance(raw, dict):
        raise ValueError("variant context must be a mapping")
    _reject_unknown(raw, ContextSelection, kind="variant context")
    if not str(raw.get("system_id") or "").strip():
        raise ValueError("variant context requires an explicit context system_id")
    system_id = validate_id(raw["system_id"], kind="context system id")
    if not str(raw.get("delivery") or "").strip():
        raise ValueError(
            f"context system {system_id} requires an explicit context delivery"
        )
    delivery = str(raw["delivery"])
    if delivery not in {"portable", "native_mcp"}:
        raise ValueError(f"unknown context delivery: {delivery}")
    return ContextSelection(
        system_id=system_id,
        delivery=delivery,
        config=_dict(raw.get("config")),
    )


def _integration_selections(raw: Any, *, kind: str) -> list[IntegrationSelection]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{kind} integrations must be a list")
    result: list[IntegrationSelection] = []
    for index, value in enumerate(raw, start=1):
        if isinstance(value, str):
            integration_id = validate_id(value, kind="integration id")
            result.append(IntegrationSelection(id=integration_id))
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{kind} integration {index} must be a string or mapping")
        unknown = sorted(set(value) - {"id", "config"})
        if unknown:
            raise ValueError(
                f"unknown {kind} integration field(s): {', '.join(unknown)}"
            )
        integration_id = validate_id(value.get("id"), kind="integration id")
        result.append(
            IntegrationSelection(id=integration_id, config=_dict(value.get("config")))
        )
    require_unique([item.id for item in result], kind=f"{kind} integration")
    return result


def _workloads(raw: Any) -> list[WorkloadSpec]:
    values = _list(raw)
    workloads: list[WorkloadSpec] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ValueError("workload must be a mapping")
        _reject_unknown(value, WorkloadSpec, kind="workload")
        workload_id = validate_id(
            value.get("id") or f"workload-{index}", kind="workload id"
        )
        runner = str(value.get("runner") or "harbor")
        if runner not in {"harbor", "retrieval", "sequence"}:
            raise ValueError(f"unknown workload runner: {runner}")
        required_capabilities = sorted(
            context_capabilities(
                _string_list(value.get("required_capabilities")),
                kind=f"workload {workload_id}",
            )
        )
        workloads.append(
            WorkloadSpec(
                id=workload_id,
                runner=runner,
                manifest=Path(str(value["manifest"]))
                if value.get("manifest")
                else None,
                dataset=_optional_str(value.get("dataset")),
                systems=_string_list(value.get("systems")),
                variants=_string_list(value.get("variants")),
                harness_assignment=_harness_assignment(
                    value.get("harness_assignment"), workload_id
                ),
                required_capabilities=required_capabilities,
                n_tasks=_positive_int(
                    value.get("n_tasks"), kind=f"workload {workload_id} n_tasks"
                ),
                n_attempts=_positive_int(
                    value.get("n_attempts"),
                    kind=f"workload {workload_id} n_attempts",
                ),
                artifacts=_list(value.get("artifacts")),
                scorers=_scorer_refs(value.get("scorers"), workload_id),
            )
        )
    return workloads


def _harness_assignment(value: Any, workload_id: str) -> str:
    assignment = str(value or "cross")
    if assignment not in {"cross", "latin_square"}:
        raise ValueError(
            f"workload {workload_id} harness_assignment must be cross or latin_square"
        )
    return assignment


def _evaluation_generation(raw: Any) -> EvaluationGenerationSpec | None:
    if raw in (None, ""):
        return None
    if not isinstance(raw, dict):
        raise ValueError("evaluation_generation must be a mapping")
    _reject_unknown(raw, EvaluationGenerationSpec, kind="evaluation generation")
    size = _positive_int(raw.get("size", 8), kind="evaluation generation size")
    assert size is not None
    sources: list[EvaluationSourceSpec] = []
    for index, value in enumerate(_list(raw.get("sources")), start=1):
        if not isinstance(value, dict):
            raise ValueError("evaluation source must be a mapping")
        _reject_unknown(value, EvaluationSourceSpec, kind="evaluation source")
        kind = str(value.get("kind") or "").strip()
        if kind not in {"seed", "file", "mcp"}:
            raise ValueError(
                f"evaluation source {index} kind must be seed, file, or mcp"
            )
        path = _optional_str(value.get("path"))
        text = _optional_str(value.get("text"))
        server = _optional_str(value.get("server"))
        tools = _string_list(value.get("tools"))
        resources = _string_list(value.get("resources"))
        if kind == "seed" and not text:
            raise ValueError(f"evaluation source {index} seed text is required")
        if kind == "file":
            if not path:
                raise ValueError(f"evaluation source {index} file path is required")
            _safe_repo_path(path, kind=f"evaluation source {index} path")
        if kind == "mcp" and not server:
            raise ValueError(f"evaluation source {index} MCP server is required")
        if server:
            validate_id(server, kind=f"evaluation source {index} MCP server")
        sources.append(
            EvaluationSourceSpec(
                kind=kind,
                path=path,
                text=text,
                server=server,
                tools=tools,
                resources=resources,
            )
        )
    suite_id = validate_id(raw.get("suite_id"), kind="evaluation suite id")
    workload_id = validate_id(raw.get("workload_id"), kind="evaluation workload id")
    return EvaluationGenerationSpec(
        suite_id=suite_id,
        workload_id=workload_id,
        size=size,
        sources=sources,
    )


def research_view_from_data(raw: Any) -> ExperimentResearchViewSpec | None:
    if raw in (None, ""):
        return None
    if not isinstance(raw, dict):
        raise ValueError("research_view must be a mapping")
    _reject_unknown(raw, ExperimentResearchViewSpec, kind="research view")
    scorers: list[ResearchScorerSpec] = []
    for index, value in enumerate(_list(raw.get("scorers")), start=1):
        if not isinstance(value, dict):
            raise ValueError(f"research view scorer {index} must be a mapping")
        _reject_unknown(value, ResearchScorerSpec, kind="research view scorer")
        scorer_kind = str(value.get("kind") or "")
        if scorer_kind not in {
            "benchmark",
            "deterministic",
            "criteria",
            "llm_judge",
        }:
            raise ValueError(f"unknown research view scorer kind: {scorer_kind}")
        dimensions: list[ResearchScoreDimensionSpec] = []
        for dimension_index, dimension in enumerate(
            _list(value.get("dimensions")), start=1
        ):
            if not isinstance(dimension, dict):
                raise ValueError(
                    f"research view score dimension {dimension_index} "
                    "must be a mapping"
                )
            _reject_unknown(
                dimension,
                ResearchScoreDimensionSpec,
                kind="research view score dimension",
            )
            target = dimension.get("target")
            if target is not None and not isinstance(
                target, str | int | float | bool
            ):
                raise ValueError("research view score target must be scalar")
            dimension_id = validate_id(
                dimension.get("id"),
                kind="research view score dimension id",
            )
            dimension_label = str(dimension.get("label") or "").strip()
            if not dimension_label:
                raise ValueError(
                    f"research view score dimension {dimension_id} requires a label"
                )
            dimensions.append(
                ResearchScoreDimensionSpec(
                    id=dimension_id,
                    label=dimension_label,
                    description=str(dimension.get("description") or "").strip(),
                    source_key=_optional_str(dimension.get("source_key")),
                    target=target,
                    primary=bool(dimension.get("primary", False)),
                )
            )
        scorer_id = validate_id(value.get("id"), kind="research view scorer id")
        label = str(value.get("label") or "").strip()
        description = str(value.get("description") or "").strip()
        if not label or not description:
            raise ValueError(
                f"research view scorer {scorer_id} requires label and description"
            )
        threshold = (
            float(value["threshold"])
            if value.get("threshold") is not None
            else None
        )
        if threshold is not None and (
            not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError(
                f"research view scorer {scorer_id} threshold must be in [0, 1]"
            )
        scorers.append(
            ResearchScorerSpec(
                id=scorer_id,
                label=label,
                kind=scorer_kind,  # type: ignore[arg-type]
                description=description,
                required=bool(value.get("required", True)),
                threshold=threshold,
                aggregation=str(value.get("aggregation") or "").strip(),
                evidence_inputs=_string_list(value.get("evidence_inputs")),
                revision=_optional_str(value.get("revision")),
                model=_optional_str(value.get("model")),
                rubric_summary=str(value.get("rubric_summary") or "").strip(),
                blind_fields=_string_list(value.get("blind_fields")),
                dimensions=dimensions,
            )
        )
    treatment_summaries = _dict(raw.get("treatment_summaries"))
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in treatment_summaries.items()
    ):
        raise ValueError("research view treatment summaries must be strings")
    return ExperimentResearchViewSpec(
        observation=str(raw.get("observation") or "").strip(),
        rationale=str(raw.get("rationale") or "").strip(),
        alternative_explanations=_string_list(raw.get("alternative_explanations")),
        success_definition=str(raw.get("success_definition") or "").strip(),
        task_title=str(raw.get("task_title") or "").strip(),
        task_summary=str(raw.get("task_summary") or "").strip(),
        interaction_mode=str(raw.get("interaction_mode") or "").strip(),
        tools=_string_list(raw.get("tools")),
        resources=_string_list(raw.get("resources")),
        base_instruction_summary=str(
            raw.get("base_instruction_summary") or ""
        ).strip(),
        treatment_summaries={
            str(key): str(value) for key, value in treatment_summaries.items()
        },
        pass_rule=str(raw.get("pass_rule") or "").strip(),
        scorers=scorers,
    )


def _scorer_refs(value: Any, workload_id: str) -> list[ScorerSelection]:
    values = _list(value)
    result: list[ScorerSelection] = []
    for index, raw in enumerate(values, start=1):
        if not isinstance(raw, dict):
            raise ValueError(
                f"workload {workload_id} scorer {index} must be a typed mapping"
            )
        scorer_type = str(raw.get("type") or "")
        if scorer_type == "builtin":
            unknown = sorted(set(raw) - {"type", "id"})
            scorer_id = str(raw.get("id") or "")
            if unknown or scorer_id != "harbor-outcome":
                raise ValueError(
                    f"workload {workload_id} builtin scorer must be harbor-outcome"
                )
            result.append(BuiltinScorerSelection(type="builtin", id="harbor-outcome"))
            continue
        if scorer_type != "rubric" or sorted(set(raw) - {"type", "path"}):
            raise ValueError(
                f"workload {workload_id} scorer {index} type must be builtin or rubric"
            )
        ref = str(raw.get("path") or "")
        path = _safe_repo_path(ref, kind=f"workload {workload_id} scorer")
        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError(
                f"workload {workload_id} scorer must be a YAML file: {ref}"
            )
        prefix = Path("configs/fugue/evaluations")
        if path != prefix and prefix not in path.parents:
            raise ValueError(
                f"workload {workload_id} scorer must live under {prefix}: {ref}"
            )
        result.append(RubricScorerSelection(type="rubric", path=ref))
    refs = [scorer_reference(item) for item in result]
    require_unique(refs, kind=f"workload {workload_id} scorer")
    return result


def _safe_repo_path(value: str, *, kind: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{kind} must be a repository-relative path")
    return path


def _presets(raw: Any) -> list[PresetSpec]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        values = [{"id": key, **_dict(value)} for key, value in raw.items()]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError("presets must be a mapping or list")
    presets: list[PresetSpec] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ValueError("preset must be a mapping")
        _reject_unknown(value, PresetSpec, kind="preset")
        preset_id = validate_id(value.get("id") or f"preset-{index}", kind="preset id")
        presets.append(
            PresetSpec(
                id=preset_id,
                workloads=_string_list(value.get("workloads")),
                systems=_string_list(value.get("systems")),
                harnesses=_string_list(value.get("harnesses")),
                n_tasks=_positive_int(
                    value.get("n_tasks"), kind=f"preset {preset_id} n_tasks"
                ),
                n_attempts=_positive_int(
                    value.get("n_attempts"), kind=f"preset {preset_id} n_attempts"
                ),
                n_concurrent=_positive_int(
                    value.get("n_concurrent"),
                    kind=f"preset {preset_id} n_concurrent",
                ),
                scheduling_seed=_optional_str(value.get("scheduling_seed")),
                selection_lock_required=bool(
                    value.get("selection_lock_required", False)
                ),
                workload_overrides=_workload_overrides(
                    value.get("workload_overrides"), preset_id
                ),
            )
        )
    return presets


def _library_item(item_id: str, path: Path, title: str | None) -> LibraryItem:
    return LibraryItem(
        id=item_id,
        title=title or item_id,
        path=path.as_posix(),
        sha256=_file_sha256(path),
    )


def _list_markdown_items(root: Path) -> list[LibraryItem]:
    if not root.exists():
        return []
    return [
        _library_item(path.stem, path, _title_from_markdown(path.read_text()))
        for path in sorted(root.glob("*.md"))
        if path.is_file()
    ]


def _list_yaml_items(root: Path) -> list[LibraryItem]:
    if not root.exists():
        return []
    return [
        _library_item(path.stem, path, _experiment_title(path))
        for path in sorted(root.glob("*.yaml"))
        if path.is_file()
    ]


def _experiment_title(path: Path) -> str:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return path.stem
    return (
        str(raw.get("title") or raw.get("id") or path.stem)
        if isinstance(raw, dict)
        else path.stem
    )


def _title_from_markdown(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_text(body: str) -> str:
    text = str(body or "")
    return text if text.endswith("\n") else text + "\n"


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _trace_content(value: Any) -> str:
    selected = str(value or "full").strip().lower()
    if selected not in {"full", "metadata"}:
        raise ValueError("trace_content must be 'full' or 'metadata'")
    return selected


def _reject_unknown(raw: dict[str, Any], contract: type, *, kind: str) -> None:
    unknown = sorted(set(raw) - set(contract.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown {kind} field(s): {', '.join(unknown)}")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"expected string or list, got {type(value).__name__}")


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _positive_int(value: Any, *, kind: str) -> int | None:
    parsed = _optional_int(value)
    if parsed is not None and parsed < 1:
        raise ValueError(f"{kind} must be positive")
    return parsed


def _workload_overrides(value: Any, preset_id: str) -> dict[str, dict[str, Any]]:
    values = _dict(value)
    allowed = {"n_tasks", "n_attempts", "n_concurrent"}
    result: dict[str, dict[str, Any]] = {}
    for key, item in values.items():
        settings = _dict(item)
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise ValueError(
                f"preset {preset_id} workload {key} has unknown override(s): "
                f"{', '.join(unknown)}"
            )
        result[str(key)] = {
            name: _positive_int(
                selected,
                kind=f"preset {preset_id} workload {key} {name}",
            )
            for name, selected in settings.items()
        }
    return result


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_paths_to_strings(item) for item in value]
    return value
