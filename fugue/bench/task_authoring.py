from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.library import validate_id
from fugue.bench.manifest import BenchmarkManifest

TASK_AUTHORING_SCHEMA_VERSION = 1
TASK_PROFILE_PATH = Path("configs/fugue/task-authoring/profiles.yaml")
TASK_SUITE_RUNTIME_ROOT = Path(".fugue/runtime/campaigns")
TASK_DATASET_CACHE_ROOT = Path(".fugue/cache/authored-task-datasets")

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PARTITIONS = frozenset({"qualification", "discovery", "holdout"})
_SAFE_PROMPT_PARTS = frozenset({"text", "resource"})
_SAFE_ENVIRONMENT_KINDS = frozenset({"repository", "artifact", "live_service"})
_SAFE_INTERACTION_KINDS = frozenset({"single_turn", "scripted", "model"})
_SAFE_EVALUATORS = frozenset(
    {
        "benchmark_outcome",
        "answer_contains",
        "answer_regex",
        "artifact",
        "tool_evidence",
        "repository_diff",
        "judge",
        "inline_python",
    }
)
_SAFE_EVIDENCE = frozenset(
    {
        "answer",
        "artifacts",
        "benchmark",
        "changed_paths",
        "opened_paths",
        "tool_calls",
        "trace_summary",
    }
)
_SAFE_HARNESSES = ("hermes", "openclaw", "claude-code", "codex")
_HARNESS_AGENTS = {
    "hermes": "fugue.agents:FugueHermes",
    "openclaw": "fugue.agents:FugueOpenClaw",
    "claude-code": "fugue.agents:FugueClaudeCode",
    "codex": "fugue.agents:FugueCodex",
}
_MAX_TEXT = 16_000
_MAX_CODE = 32_000


@dataclass(frozen=True)
class TaskAuthoringLimitsV1:
    max_tasks: int
    max_scenarios: int
    max_prompt_bytes: int
    max_authored_asset_bytes: int
    max_user_turns: int
    max_agent_turns: int
    max_interactor_calls: int
    max_judge_calls: int
    scorer_timeout_sec: int
    scorer_memory_mb: int
    scorer_cpus: float
    scorer_output_bytes: int


@dataclass(frozen=True)
class TaskAuthoringPolicyV1:
    enabled_stages: tuple[str, ...]
    allowed_partitions: tuple[str, ...]
    allowed_environment_profiles: tuple[str, ...]
    allowed_resource_profiles: tuple[str, ...]
    allowed_interactor_profiles: tuple[str, ...]
    allowed_judge_profiles: tuple[str, ...]
    allowed_scorer_runtimes: tuple[str, ...]
    allowed_prompt_parts: tuple[str, ...]
    adaptive_discovery: bool
    limits: TaskAuthoringLimitsV1

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class EnvironmentProfileV1:
    id: str
    title: str
    kind: str
    base_image: str
    supported_harnesses: tuple[str, ...]
    capabilities: tuple[str, ...]
    cpus: float
    memory_mb: int
    storage_mb: int
    profile_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ResourceProfileV1:
    id: str
    title: str
    kind: str
    path: str
    sha256: str
    media_type: str
    target: str
    profile_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class InteractorProfileV1:
    id: str
    title: str
    kind: str
    model: str | None
    directions: tuple[str, ...]
    supported_harnesses: tuple[str, ...]
    profile_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class JudgeProfileV1:
    id: str
    title: str
    model: str
    prompt: str
    evidence: tuple[str, ...]
    blind_fields: tuple[str, ...]
    input_cost_per_million: float
    output_cost_per_million: float
    profile_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ScorerRuntimeProfileV1:
    id: str
    title: str
    image: str
    command: tuple[str, ...]
    profile_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class TaskProfileCatalogV1:
    schema_version: int
    environments: tuple[EnvironmentProfileV1, ...]
    resources: tuple[ResourceProfileV1, ...]
    interactors: tuple[InteractorProfileV1, ...]
    judges: tuple[JudgeProfileV1, ...]
    scorer_runtimes: tuple[ScorerRuntimeProfileV1, ...]
    source_sha256: str
    catalog_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    def safe_dict(self) -> dict[str, Any]:
        def items(values: Sequence[Any]) -> list[dict[str, Any]]:
            return [
                {
                    "id": value.id,
                    "title": value.title,
                    "kind": getattr(value, "kind", "runtime"),
                    "capabilities": list(getattr(value, "capabilities", ())),
                    "supported_harnesses": list(
                        getattr(value, "supported_harnesses", ())
                    ),
                    "profile_digest": value.profile_digest,
                }
                for value in values
            ]

        return {
            "schema_version": self.schema_version,
            "environments": items(self.environments),
            "resources": items(self.resources),
            "interactors": items(self.interactors),
            "judges": items(self.judges),
            "scorer_runtimes": items(self.scorer_runtimes),
            "source_sha256": self.source_sha256,
            "catalog_digest": self.catalog_digest,
        }

    def environment(self, profile_id: str) -> EnvironmentProfileV1:
        return _profile(self.environments, profile_id, "environment")

    def resource(self, profile_id: str) -> ResourceProfileV1:
        return _profile(self.resources, profile_id, "resource")

    def interactor(self, profile_id: str) -> InteractorProfileV1:
        return _profile(self.interactors, profile_id, "interactor")

    def judge(self, profile_id: str) -> JudgeProfileV1:
        return _profile(self.judges, profile_id, "judge")

    def scorer_runtime(self, profile_id: str) -> ScorerRuntimeProfileV1:
        return _profile(self.scorer_runtimes, profile_id, "scorer runtime")


@dataclass(frozen=True)
class PromptPartV1:
    type: str
    text: str | None = None
    resource_profile_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)))


@dataclass(frozen=True)
class TaskEnvironmentV1:
    profile_id: str
    repository: dict[str, str] | None = None
    integration_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)))


@dataclass(frozen=True)
class TaskInteractionV1:
    type: str
    profile_id: str | None
    scripted_turns: tuple[str, ...]
    directions: tuple[str, ...]
    max_user_turns: int
    max_agent_turns: int
    timeout_sec: int

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)))


@dataclass(frozen=True)
class CriterionEvaluatorV1:
    type: str
    profile_id: str | None = None
    runtime_profile_id: str | None = None
    source: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)))


@dataclass(frozen=True)
class CriterionV1:
    id: str
    description: str
    evaluator: CriterionEvaluatorV1
    evidence: tuple[str, ...]
    weight: float
    threshold: float
    required: bool

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class CriteriaSetV1:
    id: str
    title: str
    pass_threshold: float
    criteria: tuple[CriterionV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class AuthoredTaskV1:
    id: str
    title: str
    prompt: tuple[PromptPartV1, ...]
    environment: TaskEnvironmentV1
    interaction: TaskInteractionV1
    criteria_set_id: str
    tags: tuple[str, ...]
    partition: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ScenarioTaskRefV1:
    task_id: str
    weight: float
    must_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class TaskScenarioV1:
    id: str
    title: str
    tasks: tuple[ScenarioTaskRefV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class TaskSuiteDraftV1:
    schema_version: int
    id: str
    title: str
    objective: str
    stage_id: str
    tasks: tuple[AuthoredTaskV1, ...]
    scenarios: tuple[TaskScenarioV1, ...]
    criteria_sets: tuple[CriteriaSetV1, ...]
    parent_outcome_id: str | None = None
    decision_rationale: str = ""
    draft_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class TaskSuitePreviewV1:
    schema_version: int
    campaign_id: str
    catalog_digest: str
    policy_digest: str
    draft: dict[str, Any]
    task_count: int
    scenario_count: int
    prompt_bytes: int
    authored_asset_bytes: int
    estimated_calls: dict[str, int]
    capability_matrix: tuple[dict[str, Any], ...]
    component_digests: dict[str, str]
    eligible: bool
    failures: tuple[str, ...]
    preview_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class TaskSuiteLockV1:
    schema_version: int
    campaign_id: str
    suite_id: str
    stage_id: str
    catalog_digest: str
    policy_digest: str
    preview_digest: str
    parent_outcome_id: str | None
    task_definition_digest: str
    criteria_digest: str
    task_count: int
    scenario_count: int
    task_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    partitions: tuple[str, ...]
    component_digests: dict[str, str]
    manifest_path: str
    public_cases_path: str
    private_evaluation_path: str
    suite_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class TaskScoringRevisionV1:
    schema_version: int
    id: str
    evidence_view: Literal["answer", "answer_artifacts_tools"]
    supersedes: str | None = None
    reason: str = ""
    revision_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(_json_value(asdict(self)), preserve_false=True)


@dataclass(frozen=True)
class TaskEvaluationV1:
    schema_version: int
    evaluation_id: str
    campaign_id: str
    run_id: str
    task_suite_digest: str
    criteria_digest: str
    scoring_revision: dict[str, Any]
    prediction_results: tuple[dict[str, Any], ...]
    evaluated_predictions: int
    passed: int
    failed: int
    unavailable: int
    observed_cost_usd: float
    accounted_cost_usd: float
    evaluation_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class TaskStudyAnalysisV1:
    schema_version: int
    analysis_id: str
    campaign_id: str
    run_id: str
    task_suite_digest: str
    evaluation_digest: str
    task_results: tuple[dict[str, Any], ...]
    scenario_results: tuple[dict[str, Any], ...]
    harness_results: tuple[dict[str, Any], ...]
    interaction_results: tuple[dict[str, Any], ...]
    contrasts: tuple[dict[str, Any], ...]
    scenario_interactions: tuple[dict[str, Any], ...]
    judge_sensitivity: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    analysis_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


def task_authoring_policy_from_dict(raw: Any) -> TaskAuthoringPolicyV1 | None:
    if raw in (None, False):
        return None
    value = _mapping(raw, "task authoring policy")
    _reject_unknown(
        value,
        {
            "enabled_stages",
            "allowed_partitions",
            "allowed_environment_profiles",
            "allowed_resource_profiles",
            "allowed_interactor_profiles",
            "allowed_judge_profiles",
            "allowed_scorer_runtimes",
            "allowed_prompt_parts",
            "adaptive_discovery",
            "limits",
        },
        "task authoring policy",
    )
    limits_raw = _mapping(value.get("limits"), "task authoring limits")
    _reject_unknown(
        limits_raw,
        {
            "max_tasks",
            "max_scenarios",
            "max_prompt_bytes",
            "max_authored_asset_bytes",
            "max_user_turns",
            "max_agent_turns",
            "max_interactor_calls",
            "max_judge_calls",
            "scorer_timeout_sec",
            "scorer_memory_mb",
            "scorer_cpus",
            "scorer_output_bytes",
        },
        "task authoring limits",
    )
    partitions = _text_tuple(value.get("allowed_partitions"), "task partition")
    unknown_partitions = sorted(set(partitions) - _SAFE_PARTITIONS)
    if unknown_partitions:
        raise ValueError(
            "unknown task authoring partition(s): " + ", ".join(unknown_partitions)
        )
    prompt_parts = _text_tuple(value.get("allowed_prompt_parts"), "prompt part")
    unknown_prompt = sorted(set(prompt_parts) - _SAFE_PROMPT_PARTS)
    if unknown_prompt:
        raise ValueError("unknown prompt part type(s): " + ", ".join(unknown_prompt))
    return TaskAuthoringPolicyV1(
        enabled_stages=_id_tuple(value.get("enabled_stages"), "task stage"),
        allowed_partitions=partitions,
        allowed_environment_profiles=_id_tuple(
            value.get("allowed_environment_profiles"), "environment profile"
        ),
        allowed_resource_profiles=_id_tuple(
            value.get("allowed_resource_profiles"), "resource profile", allow_empty=True
        ),
        allowed_interactor_profiles=_id_tuple(
            value.get("allowed_interactor_profiles"),
            "interactor profile",
            allow_empty=True,
        ),
        allowed_judge_profiles=_id_tuple(
            value.get("allowed_judge_profiles"), "judge profile", allow_empty=True
        ),
        allowed_scorer_runtimes=_id_tuple(
            value.get("allowed_scorer_runtimes"),
            "scorer runtime",
            allow_empty=True,
        ),
        allowed_prompt_parts=prompt_parts,
        adaptive_discovery=bool(value.get("adaptive_discovery", False)),
        limits=TaskAuthoringLimitsV1(
            max_tasks=_positive_int(limits_raw.get("max_tasks"), "max tasks"),
            max_scenarios=_positive_int(
                limits_raw.get("max_scenarios"), "max scenarios"
            ),
            max_prompt_bytes=_positive_int(
                limits_raw.get("max_prompt_bytes"), "max prompt bytes"
            ),
            max_authored_asset_bytes=_positive_int(
                limits_raw.get("max_authored_asset_bytes"),
                "max authored asset bytes",
            ),
            max_user_turns=_positive_int(
                limits_raw.get("max_user_turns"), "max user turns"
            ),
            max_agent_turns=_positive_int(
                limits_raw.get("max_agent_turns"), "max agent turns"
            ),
            max_interactor_calls=_non_negative_int(
                limits_raw.get("max_interactor_calls"), "max interactor calls"
            ),
            max_judge_calls=_non_negative_int(
                limits_raw.get("max_judge_calls"), "max judge calls"
            ),
            scorer_timeout_sec=_positive_int(
                limits_raw.get("scorer_timeout_sec"), "scorer timeout"
            ),
            scorer_memory_mb=_positive_int(
                limits_raw.get("scorer_memory_mb"), "scorer memory"
            ),
            scorer_cpus=_positive_number(limits_raw.get("scorer_cpus"), "scorer cpus"),
            scorer_output_bytes=_positive_int(
                limits_raw.get("scorer_output_bytes"), "scorer output bytes"
            ),
        ),
    )


def load_task_profiles(repo_root: Path | None = None) -> TaskProfileCatalogV1:
    root = (repo_root or Path.cwd()).resolve()
    path = root / TASK_PROFILE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"task authoring profiles not found: {path}")
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: task authoring profiles must be a mapping")
    return task_profile_catalog_from_dict(
        raw, source_sha256=hashlib.sha256(text.encode()).hexdigest()
    )


def task_profile_catalog_from_dict(
    raw: Mapping[str, Any], *, source_sha256: str | None = None
) -> TaskProfileCatalogV1:
    _reject_unknown(
        raw,
        {
            "schema_version",
            "environments",
            "resources",
            "interactors",
            "judges",
            "scorer_runtimes",
            "source_sha256",
            "catalog_digest",
        },
        "task profile catalog",
    )
    _schema(raw, "task profile catalog")
    source_digest = source_sha256 or _required_digest(
        raw.get("source_sha256"), "profile source sha256"
    )
    catalog = TaskProfileCatalogV1(
        schema_version=TASK_AUTHORING_SCHEMA_VERSION,
        environments=tuple(
            _environment_profile(item)
            for item in _sequence(raw.get("environments"), "environment profiles")
        ),
        resources=tuple(
            _resource_profile(item)
            for item in _sequence(raw.get("resources"), "resource profiles")
        ),
        interactors=tuple(
            _interactor_profile(item)
            for item in _sequence(raw.get("interactors"), "interactor profiles")
        ),
        judges=tuple(
            _judge_profile(item)
            for item in _sequence(raw.get("judges"), "judge profiles")
        ),
        scorer_runtimes=tuple(
            _scorer_runtime_profile(item)
            for item in _sequence(raw.get("scorer_runtimes"), "scorer runtimes")
        ),
        source_sha256=_required_digest(source_digest, "profile source sha256"),
    )
    for label, values in (
        ("environment", catalog.environments),
        ("resource", catalog.resources),
        ("interactor", catalog.interactors),
        ("judge", catalog.judges),
        ("scorer runtime", catalog.scorer_runtimes),
    ):
        _require_unique([item.id for item in values], label)
    digest = _artifact_digest(catalog.to_dict(), "catalog_digest")
    supplied = str(raw.get("catalog_digest") or "")
    if supplied and supplied != digest:
        raise ValueError("task profile catalog digest does not match")
    return replace(catalog, catalog_digest=digest)


def task_suite_draft_from_dict(
    raw: Mapping[str, Any], *, require_digest: bool = False
) -> TaskSuiteDraftV1:
    _reject_unknown(
        raw,
        {
            "schema_version",
            "id",
            "title",
            "objective",
            "stage_id",
            "tasks",
            "scenarios",
            "criteria_sets",
            "parent_outcome_id",
            "decision_rationale",
            "draft_digest",
        },
        "task suite draft",
    )
    _schema(raw, "task suite draft")
    draft = TaskSuiteDraftV1(
        schema_version=TASK_AUTHORING_SCHEMA_VERSION,
        id=validate_id(raw.get("id") or "", kind="task suite id"),
        title=_bounded_text(raw.get("title"), "task suite title", 200),
        objective=_bounded_text(raw.get("objective"), "task suite objective", 4000),
        stage_id=validate_id(raw.get("stage_id") or "", kind="campaign stage id"),
        tasks=tuple(
            _authored_task(item)
            for item in _sequence(raw.get("tasks"), "authored tasks")
        ),
        scenarios=tuple(
            _scenario(item) for item in _sequence(raw.get("scenarios"), "scenarios")
        ),
        criteria_sets=tuple(
            _criteria_set(item)
            for item in _sequence(raw.get("criteria_sets"), "criteria sets")
        ),
        parent_outcome_id=(
            validate_id(raw["parent_outcome_id"], kind="outcome id")
            if raw.get("parent_outcome_id")
            else None
        ),
        decision_rationale=(
            _bounded_text(raw.get("decision_rationale"), "decision rationale", 4000)
            if raw.get("decision_rationale")
            else ""
        ),
        draft_digest=str(raw.get("draft_digest") or ""),
    )
    _validate_draft_structure(draft)
    digest = _artifact_digest(draft.to_dict(), "draft_digest")
    if require_digest and draft.draft_digest != digest:
        raise ValueError("draft_digest does not match the task suite draft")
    if draft.draft_digest and draft.draft_digest != digest:
        raise ValueError("draft_digest does not match the task suite draft")
    return replace(draft, draft_digest=digest)


def task_suite_preview_from_dict(raw: Mapping[str, Any]) -> TaskSuitePreviewV1:
    fields = {field.name for field in TaskSuitePreviewV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "task suite preview")
    value = TaskSuitePreviewV1(
        schema_version=_schema(raw, "task suite preview"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        catalog_digest=_required_digest(raw.get("catalog_digest"), "catalog digest"),
        policy_digest=_required_digest(raw.get("policy_digest"), "policy digest"),
        draft=_mapping(raw.get("draft"), "task suite draft"),
        task_count=_positive_int(raw.get("task_count"), "task count"),
        scenario_count=_positive_int(raw.get("scenario_count"), "scenario count"),
        prompt_bytes=_non_negative_int(raw.get("prompt_bytes"), "prompt bytes"),
        authored_asset_bytes=_non_negative_int(
            raw.get("authored_asset_bytes"), "authored asset bytes"
        ),
        estimated_calls={
            str(key): _non_negative_int(value, f"estimated {key} calls")
            for key, value in _mapping(
                raw.get("estimated_calls"), "estimated calls"
            ).items()
        },
        capability_matrix=tuple(
            _mapping(item, "capability coordinate")
            for item in _sequence(raw.get("capability_matrix"), "capability matrix")
        ),
        component_digests=_digest_mapping(
            raw.get("component_digests"), "component digests"
        ),
        eligible=bool(raw.get("eligible")),
        failures=_text_tuple(raw.get("failures"), "failure", allow_empty=True),
        preview_digest=_required_digest(raw.get("preview_digest"), "preview digest"),
    )
    _verify_artifact(value.to_dict(), "preview_digest", "task suite preview")
    task_suite_draft_from_dict(value.draft, require_digest=True)
    return value


def task_suite_lock_from_dict(raw: Mapping[str, Any]) -> TaskSuiteLockV1:
    fields = {field.name for field in TaskSuiteLockV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "task suite lock")
    value = TaskSuiteLockV1(
        schema_version=_schema(raw, "task suite lock"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        suite_id=validate_id(raw.get("suite_id") or "", kind="task suite id"),
        stage_id=validate_id(raw.get("stage_id") or "", kind="campaign stage id"),
        catalog_digest=_required_digest(raw.get("catalog_digest"), "catalog digest"),
        policy_digest=_required_digest(raw.get("policy_digest"), "policy digest"),
        preview_digest=_required_digest(raw.get("preview_digest"), "preview digest"),
        parent_outcome_id=(
            validate_id(raw["parent_outcome_id"], kind="outcome id")
            if raw.get("parent_outcome_id")
            else None
        ),
        task_definition_digest=_required_digest(
            raw.get("task_definition_digest"), "task definition digest"
        ),
        criteria_digest=_required_digest(raw.get("criteria_digest"), "criteria digest"),
        task_count=_positive_int(raw.get("task_count"), "task count"),
        scenario_count=_positive_int(raw.get("scenario_count"), "scenario count"),
        task_ids=_id_tuple(raw.get("task_ids"), "task"),
        scenario_ids=_id_tuple(raw.get("scenario_ids"), "scenario"),
        partitions=_text_tuple(raw.get("partitions"), "partition"),
        component_digests=_digest_mapping(
            raw.get("component_digests"), "component digests"
        ),
        manifest_path=_safe_relative_path(raw.get("manifest_path"), "manifest path"),
        public_cases_path=_safe_relative_path(
            raw.get("public_cases_path"), "public cases path"
        ),
        private_evaluation_path=_safe_relative_path(
            raw.get("private_evaluation_path"), "private evaluation path"
        ),
        suite_digest=_required_digest(raw.get("suite_digest"), "suite digest"),
    )
    _verify_artifact(value.to_dict(), "suite_digest", "task suite lock")
    return value


def scoring_revision_from_dict(raw: Mapping[str, Any]) -> TaskScoringRevisionV1:
    _reject_unknown(
        raw,
        {
            "schema_version",
            "id",
            "evidence_view",
            "supersedes",
            "reason",
            "revision_digest",
        },
        "task scoring revision",
    )
    _schema(raw, "task scoring revision")
    evidence_view = str(raw.get("evidence_view") or "")
    if evidence_view not in {"answer", "answer_artifacts_tools"}:
        raise ValueError(
            "scoring evidence_view must be answer or answer_artifacts_tools"
        )
    revision = TaskScoringRevisionV1(
        schema_version=TASK_AUTHORING_SCHEMA_VERSION,
        id=validate_id(raw.get("id") or "", kind="scoring revision id"),
        evidence_view=evidence_view,  # type: ignore[arg-type]
        supersedes=(
            _required_digest(raw["supersedes"], "superseded evaluation digest")
            if raw.get("supersedes")
            else None
        ),
        reason=(
            _bounded_text(raw.get("reason"), "scoring revision reason", 1000)
            if raw.get("reason")
            else ""
        ),
        revision_digest=str(raw.get("revision_digest") or ""),
    )
    digest = _artifact_digest(revision.to_dict(), "revision_digest")
    if revision.revision_digest and revision.revision_digest != digest:
        raise ValueError("revision_digest does not match scoring revision")
    return replace(revision, revision_digest=digest)


def task_evaluation_from_dict(raw: Mapping[str, Any]) -> TaskEvaluationV1:
    fields = {field.name for field in TaskEvaluationV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "task evaluation")
    value = TaskEvaluationV1(
        schema_version=_schema(raw, "task evaluation"),
        evaluation_id=validate_id(raw.get("evaluation_id") or "", kind="evaluation id"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        run_id=validate_id(raw.get("run_id") or "", kind="run id"),
        task_suite_digest=_required_digest(
            raw.get("task_suite_digest"), "task suite digest"
        ),
        criteria_digest=_required_digest(raw.get("criteria_digest"), "criteria digest"),
        scoring_revision=_mapping(raw.get("scoring_revision"), "scoring revision"),
        prediction_results=tuple(
            _mapping(item, "prediction result")
            for item in _sequence(raw.get("prediction_results"), "prediction results")
        ),
        evaluated_predictions=_non_negative_int(
            raw.get("evaluated_predictions"), "evaluated predictions"
        ),
        passed=_non_negative_int(raw.get("passed"), "passed"),
        failed=_non_negative_int(raw.get("failed"), "failed"),
        unavailable=_non_negative_int(raw.get("unavailable"), "unavailable"),
        observed_cost_usd=_non_negative_number(
            raw.get("observed_cost_usd"), "observed cost"
        ),
        accounted_cost_usd=_non_negative_number(
            raw.get("accounted_cost_usd"), "accounted cost"
        ),
        evaluation_digest=_required_digest(
            raw.get("evaluation_digest"), "evaluation digest"
        ),
    )
    _verify_artifact(value.to_dict(), "evaluation_digest", "task evaluation")
    scoring_revision_from_dict(value.scoring_revision)
    return value


def task_study_analysis_from_dict(raw: Mapping[str, Any]) -> TaskStudyAnalysisV1:
    fields = {field.name for field in TaskStudyAnalysisV1.__dataclass_fields__.values()}
    _reject_unknown(raw, fields, "task study analysis")
    value = TaskStudyAnalysisV1(
        schema_version=_schema(raw, "task study analysis"),
        analysis_id=validate_id(raw.get("analysis_id") or "", kind="analysis id"),
        campaign_id=validate_id(raw.get("campaign_id") or "", kind="campaign id"),
        run_id=validate_id(raw.get("run_id") or "", kind="run id"),
        task_suite_digest=_required_digest(
            raw.get("task_suite_digest"), "task suite digest"
        ),
        evaluation_digest=_required_digest(
            raw.get("evaluation_digest"), "evaluation digest"
        ),
        task_results=_mapping_tuple(raw.get("task_results"), "task results"),
        scenario_results=_mapping_tuple(
            raw.get("scenario_results"), "scenario results"
        ),
        harness_results=_mapping_tuple(raw.get("harness_results"), "harness results"),
        interaction_results=_mapping_tuple(
            raw.get("interaction_results"), "interaction results"
        ),
        contrasts=_mapping_tuple(raw.get("contrasts"), "contrasts"),
        scenario_interactions=_mapping_tuple(
            raw.get("scenario_interactions"), "scenario interactions"
        ),
        judge_sensitivity=_mapping_tuple(
            raw.get("judge_sensitivity"), "judge sensitivity"
        ),
        limitations=_text_tuple(raw.get("limitations"), "limitation", allow_empty=True),
        analysis_digest=_required_digest(raw.get("analysis_digest"), "analysis digest"),
    )
    _verify_artifact(value.to_dict(), "analysis_digest", "task study analysis")
    return value


def preview_task_suite(
    *,
    campaign_id: str,
    catalog_digest: str,
    policy_digest: str,
    draft: TaskSuiteDraftV1,
    policy: TaskAuthoringPolicyV1,
    profiles: TaskProfileCatalogV1,
    harnesses: Sequence[str],
    repo_root: Path,
) -> TaskSuitePreviewV1:
    failures: list[str] = []
    if draft.stage_id not in policy.enabled_stages:
        failures.append(f"task authoring is not enabled for stage {draft.stage_id}")
    if len(draft.tasks) > policy.limits.max_tasks:
        failures.append("task suite exceeds the campaign task limit")
    if len(draft.scenarios) > policy.limits.max_scenarios:
        failures.append("task suite exceeds the campaign scenario limit")
    if draft.parent_outcome_id and not draft.decision_rationale:
        failures.append("adaptive task suites require a decision rationale")
    if draft.parent_outcome_id and not policy.adaptive_discovery:
        failures.append("campaign policy does not allow adaptive task authoring")

    profile_components: dict[str, str] = {"task_profiles": profiles.catalog_digest}
    prompt_bytes = 0
    asset_bytes = 0
    estimated = {"agent": len(draft.tasks), "interactor": 0, "judge": 0, "scorer": 0}
    capability_matrix: list[dict[str, Any]] = []
    criteria = {item.id: item for item in draft.criteria_sets}

    for task in draft.tasks:
        result = _preview_task(
            task,
            criterion_set=criteria[task.criteria_set_id],
            policy=policy,
            profiles=profiles,
            harnesses=harnesses,
            repo_root=repo_root,
            failures=failures,
        )
        prompt_bytes += result["prompt_bytes"]
        asset_bytes += result["asset_bytes"]
        for key in estimated:
            estimated[key] += result["estimated_calls"].get(key, 0)
        profile_components.update(result["component_digests"])
        capability_matrix.extend(result["capability_matrix"])

    if prompt_bytes > policy.limits.max_prompt_bytes:
        failures.append("task suite exceeds the campaign prompt-byte limit")
    if asset_bytes > policy.limits.max_authored_asset_bytes:
        failures.append("task suite exceeds the campaign authored-asset limit")
    if estimated["interactor"] > policy.limits.max_interactor_calls:
        failures.append("task suite exceeds the campaign interactor-call limit")
    if estimated["judge"] > policy.limits.max_judge_calls:
        failures.append("task suite exceeds the campaign judge-call limit")

    unsigned = TaskSuitePreviewV1(
        schema_version=TASK_AUTHORING_SCHEMA_VERSION,
        campaign_id=validate_id(campaign_id, kind="campaign id"),
        catalog_digest=_required_digest(catalog_digest, "catalog digest"),
        policy_digest=_required_digest(policy_digest, "policy digest"),
        draft=draft.to_dict(),
        task_count=len(draft.tasks),
        scenario_count=len(draft.scenarios),
        prompt_bytes=prompt_bytes,
        authored_asset_bytes=asset_bytes,
        estimated_calls=estimated,
        capability_matrix=tuple(capability_matrix),
        component_digests=dict(sorted(profile_components.items())),
        eligible=not failures,
        failures=tuple(dict.fromkeys(failures)),
    )
    return replace(
        unsigned,
        preview_digest=_artifact_digest(unsigned.to_dict(), "preview_digest"),
    )


def materialize_task_suite_lock(
    preview: TaskSuitePreviewV1,
    *,
    profiles: TaskProfileCatalogV1,
    repo_root: Path,
    destination: Path,
    harnesses: Sequence[str],
) -> TaskSuiteLockV1:
    _verify_artifact(preview.to_dict(), "preview_digest", "task suite preview")
    if not preview.eligible:
        raise ValueError("ineligible task suite preview cannot be locked")
    draft = task_suite_draft_from_dict(preview.draft, require_digest=True)
    definition = {
        "id": draft.id,
        "stage_id": draft.stage_id,
        "tasks": [
            {
                "id": task.id,
                "title": task.title,
                "prompt": [item.to_dict() for item in task.prompt],
                "environment": task.environment.to_dict(),
                "interaction": task.interaction.to_dict(),
                "tags": list(task.tags),
                "partition": task.partition,
            }
            for task in draft.tasks
        ],
        "scenarios": [item.to_dict() for item in draft.scenarios],
    }
    task_definition_digest = stable_digest(definition)
    private_evaluation = {
        "schema_version": TASK_AUTHORING_SCHEMA_VERSION,
        "suite_id": draft.id,
        "task_definition_digest": task_definition_digest,
        "criteria_sets": [item.to_dict() for item in draft.criteria_sets],
        "task_criteria": {task.id: task.criteria_set_id for task in draft.tasks},
        "scenarios": [item.to_dict() for item in draft.scenarios],
    }
    criteria_digest = stable_digest(private_evaluation)
    public_rows, resource_files = _public_cases(
        draft,
        profiles,
        repo_root,
        task_definition_digest,
        locked_resource_root=(destination / "resources").relative_to(repo_root),
    )
    public_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in public_rows
    )
    public_sha = hashlib.sha256(public_text.encode()).hexdigest()
    private_text = json.dumps(private_evaluation, indent=2, sort_keys=True) + "\n"
    private_sha = hashlib.sha256(private_text.encode()).hexdigest()
    manifest = _authored_manifest(
        draft,
        public_rows,
        task_definition_digest=task_definition_digest,
        criteria_digest=criteria_digest,
        source_path=(destination / "public-cases.jsonl").relative_to(repo_root),
        source_sha256=public_sha,
        harnesses=harnesses,
    )
    manifest_text = yaml.safe_dump(manifest, sort_keys=False)
    manifest_sha = hashlib.sha256(manifest_text.encode()).hexdigest()
    component_digests = {
        **preview.component_digests,
        "public_cases": public_sha,
        "private_evaluation": private_sha,
        "compiled_manifest": manifest_sha,
    }
    manifest_path = (destination / "manifest.yaml").relative_to(repo_root).as_posix()
    public_path = (destination / "public-cases.jsonl").relative_to(repo_root).as_posix()
    private_path = (
        (destination / "private-evaluation.json").relative_to(repo_root).as_posix()
    )
    unsigned = TaskSuiteLockV1(
        schema_version=TASK_AUTHORING_SCHEMA_VERSION,
        campaign_id=preview.campaign_id,
        suite_id=draft.id,
        stage_id=draft.stage_id,
        catalog_digest=preview.catalog_digest,
        policy_digest=preview.policy_digest,
        preview_digest=preview.preview_digest,
        parent_outcome_id=draft.parent_outcome_id,
        task_definition_digest=task_definition_digest,
        criteria_digest=criteria_digest,
        task_count=len(draft.tasks),
        scenario_count=len(draft.scenarios),
        task_ids=tuple(task.id for task in draft.tasks),
        scenario_ids=tuple(item.id for item in draft.scenarios),
        partitions=tuple(sorted({task.partition for task in draft.tasks})),
        component_digests=dict(sorted(component_digests.items())),
        manifest_path=manifest_path,
        public_cases_path=public_path,
        private_evaluation_path=private_path,
    )
    lock = replace(
        unsigned,
        suite_digest=_artifact_digest(unsigned.to_dict(), "suite_digest"),
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "public-cases.jsonl").write_text(public_text, encoding="utf-8")
    (destination / "private-evaluation.json").write_text(private_text, encoding="utf-8")
    (destination / "manifest.yaml").write_text(manifest_text, encoding="utf-8")
    resource_root = destination / "resources"
    for source, relative, digest in resource_files:
        target = resource_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError("locked task resource checksum changed while copying")
    atomic_write_json(destination / "task-suite-lock.json", lock.to_dict())
    return lock


def read_task_suite_lock(
    repo_root: Path, campaign_id: str, suite_digest: str
) -> TaskSuiteLockV1:
    _required_digest(suite_digest, "task suite digest")
    path = (
        repo_root
        / TASK_SUITE_RUNTIME_ROOT
        / validate_id(campaign_id, kind="campaign id")
        / "task-suites"
        / f"{suite_digest}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"task suite lock not found: {suite_digest}")
    lock = task_suite_lock_from_dict(json.loads(path.read_text(encoding="utf-8")))
    if lock.campaign_id != campaign_id or lock.suite_digest != suite_digest:
        raise ValueError("task suite lock identity does not match its path")
    verify_task_suite_lock(repo_root, lock)
    return lock


def verify_task_suite_lock(repo_root: Path, lock: TaskSuiteLockV1) -> None:
    for relative_path, component in (
        (lock.manifest_path, "compiled_manifest"),
        (lock.public_cases_path, "public_cases"),
        (lock.private_evaluation_path, "private_evaluation"),
    ):
        path = repo_root / relative_path
        if not path.is_file():
            raise ValueError(f"task suite lock asset is missing: {relative_path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != lock.component_digests[component]:
            raise ValueError(f"task suite lock asset changed: {relative_path}")


def task_suite_lock_dir(repo_root: Path, campaign_id: str, suite_digest: str) -> Path:
    return (
        repo_root
        / TASK_SUITE_RUNTIME_ROOT
        / validate_id(campaign_id, kind="campaign id")
        / "task-suite-assets"
        / _required_digest(suite_digest, "task suite digest")
    )


class AuthoredTaskMaterializer:
    def materialize(
        self,
        manifest: BenchmarkManifest,
        destination: Path,
        source_path: Path,
        *,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        root = (repo_root or Path.cwd()).resolve()
        rows = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != len(manifest.tasks):
            raise ValueError("authored task manifest count does not match its lock")
        for task in manifest.tasks:
            index = task.metadata.get("source_index")
            if not isinstance(index, int) or not 0 <= index < len(rows):
                raise ValueError(f"{task.id}: invalid authored task source_index")
            case = rows[index]
            if case.get("id") != task.id:
                raise ValueError(f"{task.id}: authored task identity drift")
            _write_authored_harbor_task(destination / task.id, case, root)
        return {
            "tasks": len(rows),
            "task_definition_digest": (
                manifest.tasks[0].metadata.get("task_authoring") or {}
            ).get("task_definition_digest"),
        }


def evaluate_task_rows(
    *,
    campaign_id: str,
    run_id: str,
    lock: TaskSuiteLockV1,
    revision: TaskScoringRevisionV1,
    rows: Sequence[Mapping[str, Any]],
    profiles: TaskProfileCatalogV1,
    repo_root: Path,
    env: Mapping[str, str],
    judge_request: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None = None,
    inline_runner: Callable[..., dict[str, Any]] | None = None,
) -> TaskEvaluationV1:
    verify_task_suite_lock(repo_root, lock)
    private = json.loads((repo_root / lock.private_evaluation_path).read_text())
    criteria_sets = {
        item.id: item
        for item in (
            _criteria_set(value) for value in private.get("criteria_sets") or []
        )
    }
    task_criteria = dict(private.get("task_criteria") or {})
    scenarios = _task_scenario_map(private.get("scenarios") or [])
    results: list[dict[str, Any]] = []
    observed_cost = 0.0
    unavailable = 0
    passed = 0
    failed = 0
    for row in rows:
        task_id = str(row.get("task_name") or row.get("task_id") or "")
        if task_id not in task_criteria:
            raise ValueError(f"prediction references unknown authored task: {task_id}")
        criterion_set = criteria_sets[str(task_criteria[task_id])]
        criterion_results: list[dict[str, Any]] = []
        for criterion in criterion_set.criteria:
            result = _evaluate_criterion(
                criterion,
                row=row,
                revision=revision,
                profiles=profiles,
                policy_evidence=set(criterion.evidence),
                repo_root=repo_root,
                env=env,
                judge_request=judge_request,
                inline_runner=inline_runner,
            )
            criterion_results.append(result)
            if result.get("cost_usd") is not None:
                observed_cost += float(result["cost_usd"])
        status, score, task_pass = _criteria_outcome(criterion_set, criterion_results)
        if status == "unavailable":
            unavailable += 1
        elif task_pass:
            passed += 1
        else:
            failed += 1
        results.append(
            {
                "prediction_id": str(row.get("prediction_id") or ""),
                "task_id": task_id,
                "scenario_id": scenarios[task_id],
                "harness": str(row.get("harness") or ""),
                "trial_index": int(row.get("trial_index") or 1),
                "interaction_type": str(
                    (row.get("task_interaction") or {}).get("type")
                    if isinstance(row.get("task_interaction"), dict)
                    else row.get("interaction_type") or "single_turn"
                ),
                "benchmark_status": row.get("status"),
                "benchmark_pass": row.get("pass"),
                "criteria_set_id": criterion_set.id,
                "criteria_status": status,
                "criteria_score": score,
                "criteria_pass": task_pass,
                "criteria": criterion_results,
                "tool_calls": row.get("tool_calls"),
                "latency_ms": row.get("latency_ms"),
                "cost_usd": row.get("cost_usd"),
            }
        )
    accounted_cost = observed_cost
    evaluation_id = validate_id(f"{run_id}-{revision.id}", kind="task evaluation id")
    unsigned = TaskEvaluationV1(
        schema_version=TASK_AUTHORING_SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        campaign_id=campaign_id,
        run_id=run_id,
        task_suite_digest=lock.suite_digest,
        criteria_digest=lock.criteria_digest,
        scoring_revision=revision.to_dict(),
        prediction_results=tuple(results),
        evaluated_predictions=len(results),
        passed=passed,
        failed=failed,
        unavailable=unavailable,
        observed_cost_usd=observed_cost,
        accounted_cost_usd=accounted_cost,
    )
    return replace(
        unsigned,
        evaluation_digest=_artifact_digest(unsigned.to_dict(), "evaluation_digest"),
    )


def analyze_task_evaluation(
    *,
    analysis_id: str,
    lock: TaskSuiteLockV1,
    evaluation: TaskEvaluationV1,
    repo_root: Path,
    bootstrap_samples: int = 2000,
) -> TaskStudyAnalysisV1:
    verify_task_suite_lock(repo_root, lock)
    private = json.loads((repo_root / lock.private_evaluation_path).read_text())
    scenarios = {
        item.id: item for item in (_scenario(value) for value in private["scenarios"])
    }
    rows = list(evaluation.prediction_results)
    task_results = _group_results(rows, ("task_id",))
    harness_results = _group_results(rows, ("harness",))
    interaction_results = _group_results(rows, ("interaction_type",))
    scenario_results = _scenario_results(rows, scenarios)
    contrasts = _aligned_contrasts(
        rows, bootstrap_samples, evaluation.evaluation_digest
    )
    interactions = _scenario_interactions(
        rows, bootstrap_samples, evaluation.evaluation_digest
    )
    judges = _judge_sensitivity(rows)
    limitations = [
        "Authored criteria and deterministic benchmark outcomes are reported separately.",
        "Harness contrasts are aligned within task and attempt; they are not a universal ranking.",
    ]
    attempts = {int(row.get("trial_index") or 1) for row in rows}
    if len(attempts) == 1:
        limitations.append("The study contains one attempt per selected coordinate.")
    unsigned = TaskStudyAnalysisV1(
        schema_version=TASK_AUTHORING_SCHEMA_VERSION,
        analysis_id=validate_id(analysis_id, kind="task analysis id"),
        campaign_id=evaluation.campaign_id,
        run_id=evaluation.run_id,
        task_suite_digest=lock.suite_digest,
        evaluation_digest=evaluation.evaluation_digest,
        task_results=tuple(task_results),
        scenario_results=tuple(scenario_results),
        harness_results=tuple(harness_results),
        interaction_results=tuple(interaction_results),
        contrasts=tuple(contrasts),
        scenario_interactions=tuple(interactions),
        judge_sensitivity=tuple(judges),
        limitations=tuple(limitations),
    )
    return replace(
        unsigned,
        analysis_digest=_artifact_digest(unsigned.to_dict(), "analysis_digest"),
    )


def run_inline_scorer(
    *,
    source: str,
    evidence: Mapping[str, Any],
    reference: Mapping[str, Any],
    profile: ScorerRuntimeProfileV1,
    limits: TaskAuthoringLimitsV1,
) -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required for isolated inline scoring")
    with tempfile.TemporaryDirectory(prefix="fugue-task-scorer-") as value:
        root = Path(value)
        (root / "scorer.py").write_text(source, encoding="utf-8")
        (root / "input.json").write_text(
            json.dumps({"evidence": evidence, "reference": reference}, sort_keys=True),
            encoding="utf-8",
        )
        command = [
            docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "32",
            "--cpus",
            str(limits.scorer_cpus),
            "--memory",
            f"{limits.scorer_memory_mb}m",
            "--mount",
            f"type=bind,src={root},dst=/input,readonly",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            profile.image,
            *profile.command,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=limits.scorer_timeout_sec,
            check=False,
            env={"PATH": os.environ.get("PATH", "")},
        )
        output = completed.stdout.encode()
        if len(output) > limits.scorer_output_bytes:
            raise ValueError("inline scorer output exceeds the configured limit")
        if completed.returncode != 0:
            raise RuntimeError(
                "inline scorer failed: " + (completed.stderr or "unknown error")[-1000:]
            )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError("inline scorer must return one JSON object")
        _scorer_payload(payload)
        return payload


def _environment_profile(raw: Any) -> EnvironmentProfileV1:
    value = _mapping(raw, "environment profile")
    _reject_unknown(
        value,
        {
            "id",
            "title",
            "kind",
            "base_image",
            "supported_harnesses",
            "capabilities",
            "cpus",
            "memory_mb",
            "storage_mb",
            "profile_digest",
        },
        "environment profile",
    )
    kind = str(value.get("kind") or "")
    if kind not in _SAFE_ENVIRONMENT_KINDS:
        raise ValueError(f"unknown task environment kind: {kind}")
    base_image = _bounded_text(value.get("base_image"), "base image", 300)
    if ":" not in base_image and "@sha256:" not in base_image:
        raise ValueError("environment base_image must be versioned")
    profile = EnvironmentProfileV1(
        id=validate_id(value.get("id") or "", kind="environment profile id"),
        title=_bounded_text(value.get("title"), "environment profile title", 200),
        kind=kind,
        base_image=base_image,
        supported_harnesses=_harness_tuple(value.get("supported_harnesses")),
        capabilities=_id_tuple(
            value.get("capabilities"), "environment capability", allow_empty=True
        ),
        cpus=_positive_number(value.get("cpus", 2), "environment cpus"),
        memory_mb=_positive_int(value.get("memory_mb", 4096), "environment memory"),
        storage_mb=_positive_int(value.get("storage_mb", 10240), "environment storage"),
        profile_digest=str(value.get("profile_digest") or ""),
    )
    return _with_profile_digest(profile, value)


def _resource_profile(raw: Any) -> ResourceProfileV1:
    value = _mapping(raw, "resource profile")
    _reject_unknown(
        value,
        {
            "id",
            "title",
            "kind",
            "path",
            "sha256",
            "media_type",
            "target",
            "profile_digest",
        },
        "resource profile",
    )
    profile = ResourceProfileV1(
        id=validate_id(value.get("id") or "", kind="resource profile id"),
        title=_bounded_text(value.get("title"), "resource title", 200),
        kind=validate_id(value.get("kind") or "file", kind="resource kind"),
        path=_safe_relative_path(value.get("path"), "resource path"),
        sha256=_required_digest(value.get("sha256"), "resource sha256"),
        media_type=_bounded_text(value.get("media_type"), "resource media type", 200),
        target=_safe_container_target(value.get("target")),
        profile_digest=str(value.get("profile_digest") or ""),
    )
    return _with_profile_digest(profile, value)


def _interactor_profile(raw: Any) -> InteractorProfileV1:
    value = _mapping(raw, "interactor profile")
    _reject_unknown(
        value,
        {
            "id",
            "title",
            "kind",
            "model",
            "directions",
            "supported_harnesses",
            "profile_digest",
        },
        "interactor profile",
    )
    kind = str(value.get("kind") or "")
    if kind not in _SAFE_INTERACTION_KINDS:
        raise ValueError(f"unknown interactor kind: {kind}")
    model = str(value.get("model") or "").strip() or None
    if kind == "model" and not model:
        raise ValueError("model interactor profile requires a model")
    profile = InteractorProfileV1(
        id=validate_id(value.get("id") or "", kind="interactor profile id"),
        title=_bounded_text(value.get("title"), "interactor title", 200),
        kind=kind,
        model=model,
        directions=_bounded_text_tuple(
            value.get("directions"), "interactor direction", allow_empty=True
        ),
        supported_harnesses=_harness_tuple(value.get("supported_harnesses")),
        profile_digest=str(value.get("profile_digest") or ""),
    )
    return _with_profile_digest(profile, value)


def _judge_profile(raw: Any) -> JudgeProfileV1:
    value = _mapping(raw, "judge profile")
    _reject_unknown(
        value,
        {
            "id",
            "title",
            "model",
            "prompt",
            "evidence",
            "blind_fields",
            "input_cost_per_million",
            "output_cost_per_million",
            "profile_digest",
        },
        "judge profile",
    )
    evidence = _text_tuple(value.get("evidence"), "judge evidence")
    unknown = sorted(set(evidence) - _SAFE_EVIDENCE)
    if unknown:
        raise ValueError("unknown judge evidence field(s): " + ", ".join(unknown))
    profile = JudgeProfileV1(
        id=validate_id(value.get("id") or "", kind="judge profile id"),
        title=_bounded_text(value.get("title"), "judge title", 200),
        model=_bounded_text(value.get("model"), "judge model", 300),
        prompt=_bounded_text(value.get("prompt"), "judge prompt", 4000),
        evidence=evidence,
        blind_fields=_id_tuple(
            value.get("blind_fields"), "judge blind field", allow_empty=True
        ),
        input_cost_per_million=_non_negative_number(
            value.get("input_cost_per_million", 0), "judge input cost"
        ),
        output_cost_per_million=_non_negative_number(
            value.get("output_cost_per_million", 0), "judge output cost"
        ),
        profile_digest=str(value.get("profile_digest") or ""),
    )
    return _with_profile_digest(profile, value)


def _scorer_runtime_profile(raw: Any) -> ScorerRuntimeProfileV1:
    value = _mapping(raw, "scorer runtime")
    _reject_unknown(
        value,
        {"id", "title", "image", "command", "profile_digest"},
        "scorer runtime",
    )
    image = _bounded_text(value.get("image"), "scorer image", 500)
    if "@sha256:" not in image or not _DIGEST.fullmatch(image.rsplit("@sha256:", 1)[1]):
        raise ValueError("scorer runtime image must use an exact sha256 digest")
    command = _bounded_text_tuple(value.get("command"), "scorer command")
    if any("/input/" not in item and item.startswith("/") for item in command):
        raise ValueError("scorer runtime command may address only /input assets")
    profile = ScorerRuntimeProfileV1(
        id=validate_id(value.get("id") or "", kind="scorer runtime id"),
        title=_bounded_text(value.get("title"), "scorer runtime title", 200),
        image=image,
        command=command,
        profile_digest=str(value.get("profile_digest") or ""),
    )
    return _with_profile_digest(profile, value)


def _authored_task(raw: Any) -> AuthoredTaskV1:
    value = _mapping(raw, "authored task")
    _reject_unknown(
        value,
        {
            "id",
            "title",
            "prompt",
            "environment",
            "interaction",
            "criteria_set_id",
            "tags",
            "partition",
        },
        "authored task",
    )
    partition = str(value.get("partition") or "")
    if partition not in _SAFE_PARTITIONS:
        raise ValueError(f"unknown task partition: {partition}")
    return AuthoredTaskV1(
        id=validate_id(value.get("id") or "", kind="authored task id"),
        title=_bounded_text(value.get("title"), "authored task title", 200),
        prompt=tuple(
            _prompt_part(item) for item in _sequence(value.get("prompt"), "task prompt")
        ),
        environment=_task_environment(value.get("environment")),
        interaction=_task_interaction(value.get("interaction")),
        criteria_set_id=validate_id(
            value.get("criteria_set_id") or "", kind="criteria set id"
        ),
        tags=_id_tuple(value.get("tags"), "task tag", allow_empty=True),
        partition=partition,
    )


def _prompt_part(raw: Any) -> PromptPartV1:
    value = _mapping(raw, "prompt part")
    _reject_unknown(value, {"type", "text", "resource_profile_id"}, "prompt part")
    part_type = str(value.get("type") or "")
    if part_type not in _SAFE_PROMPT_PARTS:
        raise ValueError(f"unknown prompt part type: {part_type}")
    text = str(value.get("text") or "").strip() or None
    resource = str(value.get("resource_profile_id") or "").strip() or None
    if part_type == "text" and (not text or resource):
        raise ValueError("text prompt parts require only text")
    if part_type == "resource" and (not resource or text):
        raise ValueError("resource prompt parts require only resource_profile_id")
    if text and len(text) > _MAX_TEXT:
        raise ValueError(f"prompt text exceeds {_MAX_TEXT} characters")
    return PromptPartV1(
        type=part_type,
        text=text,
        resource_profile_id=(
            validate_id(resource, kind="resource profile id") if resource else None
        ),
    )


def _task_environment(raw: Any) -> TaskEnvironmentV1:
    value = _mapping(raw, "task environment")
    _reject_unknown(
        value, {"profile_id", "repository", "integration_ids"}, "task environment"
    )
    repository = None
    if value.get("repository") is not None:
        repo = _mapping(value["repository"], "task repository")
        _reject_unknown(repo, {"type", "url", "commit", "path"}, "task repository")
        if str(repo.get("type") or "") != "git":
            raise ValueError("authored task repository type must be git")
        url = str(repo.get("url") or "")
        if not re.fullmatch(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", url
        ):
            raise ValueError("authored task repository must be an HTTPS GitHub URL")
        commit = str(repo.get("commit") or "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("authored task repository requires a full commit SHA")
        repository = {"type": "git", "url": url, "commit": commit}
        if repo.get("path"):
            repository["path"] = _safe_relative_path(repo["path"], "repository path")
    return TaskEnvironmentV1(
        profile_id=validate_id(
            value.get("profile_id") or "", kind="environment profile id"
        ),
        repository=repository,
        integration_ids=_id_tuple(
            value.get("integration_ids"), "integration", allow_empty=True
        ),
    )


def _task_interaction(raw: Any) -> TaskInteractionV1:
    value = _mapping(raw, "task interaction")
    _reject_unknown(
        value,
        {
            "type",
            "profile_id",
            "scripted_turns",
            "directions",
            "max_user_turns",
            "max_agent_turns",
            "timeout_sec",
        },
        "task interaction",
    )
    kind = str(value.get("type") or "single_turn")
    if kind not in _SAFE_INTERACTION_KINDS:
        raise ValueError(f"unknown task interaction type: {kind}")
    profile_id = str(value.get("profile_id") or "").strip() or None
    turns = _bounded_text_tuple(
        value.get("scripted_turns"), "scripted turn", allow_empty=True
    )
    directions = _bounded_text_tuple(
        value.get("directions"), "interaction direction", allow_empty=True
    )
    if kind == "single_turn" and (profile_id or turns or directions):
        raise ValueError("single-turn interactions cannot declare follow-up content")
    if kind == "scripted" and (not profile_id or not turns or directions):
        raise ValueError("scripted interactions require a profile and scripted_turns")
    if kind == "model" and (not profile_id or turns or not directions):
        raise ValueError("model interactions require a profile and directions")
    return TaskInteractionV1(
        type=kind,
        profile_id=(
            validate_id(profile_id, kind="interactor profile id")
            if profile_id
            else None
        ),
        scripted_turns=turns,
        directions=directions,
        max_user_turns=_positive_int(value.get("max_user_turns", 1), "max user turns"),
        max_agent_turns=_positive_int(
            value.get("max_agent_turns", 1), "max agent turns"
        ),
        timeout_sec=_positive_int(value.get("timeout_sec", 900), "interaction timeout"),
    )


def _criteria_set(raw: Any) -> CriteriaSetV1:
    value = _mapping(raw, "criteria set")
    _reject_unknown(
        value, {"id", "title", "pass_threshold", "criteria"}, "criteria set"
    )
    criteria = tuple(
        _criterion(item) for item in _sequence(value.get("criteria"), "criteria")
    )
    _require_unique([item.id for item in criteria], "criterion")
    return CriteriaSetV1(
        id=validate_id(value.get("id") or "", kind="criteria set id"),
        title=_bounded_text(value.get("title"), "criteria set title", 200),
        pass_threshold=_unit_number(
            value.get("pass_threshold"), "criteria pass threshold"
        ),
        criteria=criteria,
    )


def _criterion(raw: Any) -> CriterionV1:
    value = _mapping(raw, "criterion")
    _reject_unknown(
        value,
        {
            "id",
            "description",
            "evaluator",
            "evidence",
            "weight",
            "threshold",
            "required",
        },
        "criterion",
    )
    evaluator = _criterion_evaluator(value.get("evaluator"))
    evidence = _text_tuple(value.get("evidence"), "criterion evidence")
    unknown = sorted(set(evidence) - _SAFE_EVIDENCE)
    if unknown:
        raise ValueError("unknown criterion evidence field(s): " + ", ".join(unknown))
    return CriterionV1(
        id=validate_id(value.get("id") or "", kind="criterion id"),
        description=_bounded_text(
            value.get("description"), "criterion description", 1000
        ),
        evaluator=evaluator,
        evidence=evidence,
        weight=_positive_number(value.get("weight", 1), "criterion weight"),
        threshold=_unit_number(value.get("threshold", 1), "criterion threshold"),
        required=bool(value.get("required", False)),
    )


def _criterion_evaluator(raw: Any) -> CriterionEvaluatorV1:
    value = _mapping(raw, "criterion evaluator")
    _reject_unknown(
        value,
        {"type", "profile_id", "runtime_profile_id", "source", "config"},
        "criterion evaluator",
    )
    kind = str(value.get("type") or "")
    if kind not in _SAFE_EVALUATORS:
        raise ValueError(f"unknown criterion evaluator: {kind}")
    profile_id = str(value.get("profile_id") or "").strip() or None
    runtime_id = str(value.get("runtime_profile_id") or "").strip() or None
    source = str(value.get("source") or "").strip() or None
    config = _mapping(value.get("config", {}), "criterion evaluator config")
    _validate_evaluator_config(kind, profile_id, runtime_id, source, config)
    return CriterionEvaluatorV1(
        type=kind,
        profile_id=(
            validate_id(profile_id, kind="judge profile id") if profile_id else None
        ),
        runtime_profile_id=(
            validate_id(runtime_id, kind="scorer runtime id") if runtime_id else None
        ),
        source=source,
        config=_json_value(config),
    )


def _scenario(raw: Any) -> TaskScenarioV1:
    value = _mapping(raw, "task scenario")
    _reject_unknown(value, {"id", "title", "tasks"}, "task scenario")
    tasks = tuple(
        _scenario_task(item) for item in _sequence(value.get("tasks"), "scenario tasks")
    )
    _require_unique([item.task_id for item in tasks], "scenario task")
    return TaskScenarioV1(
        id=validate_id(value.get("id") or "", kind="scenario id"),
        title=_bounded_text(value.get("title"), "scenario title", 200),
        tasks=tasks,
    )


def _scenario_task(raw: Any) -> ScenarioTaskRefV1:
    value = _mapping(raw, "scenario task")
    _reject_unknown(value, {"task_id", "weight", "must_pass"}, "scenario task")
    return ScenarioTaskRefV1(
        task_id=validate_id(value.get("task_id") or "", kind="task id"),
        weight=_positive_number(value.get("weight", 1), "scenario task weight"),
        must_pass=bool(value.get("must_pass", False)),
    )


def _validate_draft_structure(draft: TaskSuiteDraftV1) -> None:
    _require_unique([task.id for task in draft.tasks], "authored task")
    _require_unique([item.id for item in draft.scenarios], "scenario")
    _require_unique([item.id for item in draft.criteria_sets], "criteria set")
    task_ids = {task.id for task in draft.tasks}
    criteria_ids = {item.id for item in draft.criteria_sets}
    unknown_criteria = sorted(
        {task.criteria_set_id for task in draft.tasks} - criteria_ids
    )
    if unknown_criteria:
        raise ValueError(
            "tasks reference unknown criteria set(s): " + ", ".join(unknown_criteria)
        )
    scenario_tasks = [
        item.task_id for scenario in draft.scenarios for item in scenario.tasks
    ]
    _require_unique(scenario_tasks, "scenario task membership")
    missing = sorted(task_ids - set(scenario_tasks))
    unknown = sorted(set(scenario_tasks) - task_ids)
    if missing or unknown:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unknown:
            detail.append("unknown: " + ", ".join(unknown))
        raise ValueError(
            "scenario membership must cover every task exactly once ("
            + "; ".join(detail)
            + ")"
        )


def _preview_task(
    task: AuthoredTaskV1,
    *,
    criterion_set: CriteriaSetV1,
    policy: TaskAuthoringPolicyV1,
    profiles: TaskProfileCatalogV1,
    harnesses: Sequence[str],
    repo_root: Path,
    failures: list[str],
) -> dict[str, Any]:
    if task.partition not in policy.allowed_partitions:
        failures.append(f"task {task.id} partition is not allowed")
    environment = _allowed_profile(
        task.environment.profile_id,
        policy.allowed_environment_profiles,
        profiles.environment,
        "environment",
        failures,
    )
    components: dict[str, str] = {}
    if environment is not None:
        components[f"environment:{environment.id}"] = environment.profile_digest
        _validate_environment(task, environment, failures)
    prompt_bytes, asset_bytes = _preview_prompt(
        task,
        policy=policy,
        profiles=profiles,
        repo_root=repo_root,
        components=components,
        failures=failures,
    )
    interactor = _preview_interactor(
        task,
        policy=policy,
        profiles=profiles,
        components=components,
        failures=failures,
    )
    calls = {"agent": 0, "interactor": 0, "judge": 0, "scorer": 0}
    if task.interaction.type == "model":
        calls["interactor"] = task.interaction.max_user_turns
    _preview_criteria(
        criterion_set,
        policy=policy,
        profiles=profiles,
        components=components,
        calls=calls,
        failures=failures,
    )
    capability_matrix = []
    for harness in harnesses:
        applicable, reason = _task_harness_applicability(
            task, harness, environment, interactor
        )
        capability_matrix.append(
            {
                "task_id": task.id,
                "harness": harness,
                "applicable": applicable,
                "reason": reason,
            }
        )
    return {
        "prompt_bytes": prompt_bytes,
        "asset_bytes": asset_bytes,
        "estimated_calls": calls,
        "component_digests": components,
        "capability_matrix": capability_matrix,
    }


def _preview_prompt(
    task: AuthoredTaskV1,
    *,
    policy: TaskAuthoringPolicyV1,
    profiles: TaskProfileCatalogV1,
    repo_root: Path,
    components: dict[str, str],
    failures: list[str],
) -> tuple[int, int]:
    prompt_bytes = 0
    asset_bytes = 0
    for part in task.prompt:
        if part.type not in policy.allowed_prompt_parts:
            failures.append(f"task {task.id} prompt part {part.type} is not allowed")
        if part.text:
            prompt_bytes += len(part.text.encode())
        if not part.resource_profile_id:
            continue
        resource = _allowed_profile(
            part.resource_profile_id,
            policy.allowed_resource_profiles,
            profiles.resource,
            "resource",
            failures,
        )
        if resource is None:
            continue
        components[f"resource:{resource.id}"] = resource.profile_digest
        path = repo_root / resource.path
        if not path.is_file():
            failures.append(f"resource profile {resource.id} source is missing")
            continue
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != resource.sha256:
            failures.append(f"resource profile {resource.id} checksum changed")
        asset_bytes += len(content)
    return prompt_bytes, asset_bytes


def _preview_interactor(
    task: AuthoredTaskV1,
    *,
    policy: TaskAuthoringPolicyV1,
    profiles: TaskProfileCatalogV1,
    components: dict[str, str],
    failures: list[str],
) -> InteractorProfileV1 | None:
    interactor = None
    if task.interaction.profile_id:
        interactor = _allowed_profile(
            task.interaction.profile_id,
            policy.allowed_interactor_profiles,
            profiles.interactor,
            "interactor",
            failures,
        )
        if interactor is not None:
            components[f"interactor:{interactor.id}"] = interactor.profile_digest
    _validate_interaction(task, interactor, policy, failures)
    return interactor


def _preview_criteria(
    criterion_set: CriteriaSetV1,
    *,
    policy: TaskAuthoringPolicyV1,
    profiles: TaskProfileCatalogV1,
    components: dict[str, str],
    calls: dict[str, int],
    failures: list[str],
) -> None:
    for criterion in criterion_set.criteria:
        evaluator = criterion.evaluator
        if evaluator.type == "judge" and evaluator.profile_id:
            judge = _allowed_profile(
                evaluator.profile_id,
                policy.allowed_judge_profiles,
                profiles.judge,
                "judge",
                failures,
            )
            if judge is not None:
                components[f"judge:{judge.id}"] = judge.profile_digest
            calls["judge"] += 1
        if evaluator.type == "inline_python" and evaluator.runtime_profile_id:
            runtime = _allowed_profile(
                evaluator.runtime_profile_id,
                policy.allowed_scorer_runtimes,
                profiles.scorer_runtime,
                "scorer runtime",
                failures,
            )
            if runtime is not None:
                components[f"scorer_runtime:{runtime.id}"] = runtime.profile_digest
            calls["scorer"] += 1


def _validate_environment(
    task: AuthoredTaskV1,
    profile: EnvironmentProfileV1,
    failures: list[str],
) -> None:
    if profile.kind == "repository" and not task.environment.repository:
        failures.append(
            f"task {task.id} repository environment requires a pinned repository"
        )
    if profile.kind != "repository" and task.environment.repository:
        failures.append(
            f"task {task.id} repository is incompatible with {profile.kind} environment"
        )
    if profile.kind != "live_service" and task.environment.integration_ids:
        failures.append(
            f"task {task.id} integrations require a live-service environment"
        )


def _validate_interaction(
    task: AuthoredTaskV1,
    profile: InteractorProfileV1 | None,
    policy: TaskAuthoringPolicyV1,
    failures: list[str],
) -> None:
    interaction = task.interaction
    if interaction.max_user_turns > policy.limits.max_user_turns:
        failures.append(f"task {task.id} exceeds max user turns")
    if interaction.max_agent_turns > policy.limits.max_agent_turns:
        failures.append(f"task {task.id} exceeds max agent turns")
    if interaction.type == "single_turn":
        return
    if profile is None:
        return
    if profile.kind != interaction.type:
        failures.append(
            f"task {task.id} interaction type does not match profile {profile.id}"
        )
    expected_turns = (
        len(interaction.scripted_turns)
        if interaction.type == "scripted"
        else interaction.max_user_turns
    )
    if expected_turns > interaction.max_user_turns:
        failures.append(f"task {task.id} scripted turns exceed max_user_turns")
    if interaction.max_agent_turns != expected_turns + 1:
        failures.append(
            f"task {task.id} max_agent_turns must equal initial turn plus user turns"
        )


def _task_harness_applicability(
    task: AuthoredTaskV1,
    harness: str,
    environment: EnvironmentProfileV1 | None,
    interactor: InteractorProfileV1 | None,
) -> tuple[bool, str | None]:
    if environment is not None and harness not in environment.supported_harnesses:
        return False, f"environment profile {environment.id} does not support {harness}"
    if task.interaction.type != "single_turn":
        if interactor is None or harness not in interactor.supported_harnesses:
            return (
                False,
                f"interactor profile does not qualify {harness} session resume",
            )
    return True, None


def _public_cases(
    draft: TaskSuiteDraftV1,
    profiles: TaskProfileCatalogV1,
    repo_root: Path,
    task_definition_digest: str,
    locked_resource_root: Path,
) -> tuple[list[dict[str, Any]], list[tuple[Path, Path, str]]]:
    scenario_map = {
        item.task_id: scenario.id
        for scenario in draft.scenarios
        for item in scenario.tasks
    }
    rows: list[dict[str, Any]] = []
    resources: dict[str, tuple[Path, Path, str]] = {}
    for index, task in enumerate(draft.tasks):
        environment = profiles.environment(task.environment.profile_id)
        interactor = (
            profiles.interactor(str(task.interaction.profile_id))
            if task.interaction.profile_id
            else None
        )
        harness_applicability = {
            harness: _task_harness_applicability(task, harness, environment, interactor)
            for harness in _SAFE_HARNESSES
        }
        attachments: list[dict[str, Any]] = []
        prompt_lines: list[str] = []
        for part in task.prompt:
            if part.type == "text":
                prompt_lines.append(str(part.text))
                continue
            resource = profiles.resource(str(part.resource_profile_id))
            relative = Path(resource.sha256) / Path(resource.path).name
            resources[resource.id] = (
                repo_root / resource.path,
                relative,
                resource.sha256,
            )
            attachments.append(
                {
                    "resource_profile_id": resource.id,
                    "locked_relative": (locked_resource_root / relative).as_posix(),
                    "sha256": resource.sha256,
                    "target": resource.target,
                    "media_type": resource.media_type,
                }
            )
            prompt_lines.append(
                f"Resource {resource.title} is available at {resource.target}."
            )
        rows.append(
            {
                "schema_version": TASK_AUTHORING_SCHEMA_VERSION,
                "id": task.id,
                "title": task.title,
                "instruction": "\n\n".join(prompt_lines).strip(),
                "attachments": attachments,
                "environment": {
                    "profile_id": environment.id,
                    "profile_digest": environment.profile_digest,
                    "kind": environment.kind,
                    "base_image": environment.base_image,
                    "cpus": environment.cpus,
                    "memory_mb": environment.memory_mb,
                    "storage_mb": environment.storage_mb,
                    "repository": task.environment.repository,
                    "integration_ids": list(task.environment.integration_ids),
                },
                "interaction": task.interaction.to_dict(),
                "harness_applicability": {
                    harness: {"applicable": applicable, "reason": reason}
                    for harness, (applicable, reason) in harness_applicability.items()
                },
                "profile_digests": {
                    "environment": environment.profile_digest,
                    **(
                        {"interactor": interactor.profile_digest}
                        if interactor is not None
                        else {}
                    ),
                    **{
                        f"resource:{attachment['resource_profile_id']}": profiles.resource(
                            str(attachment["resource_profile_id"])
                        ).profile_digest
                        for attachment in attachments
                    },
                },
                "scenario_id": scenario_map[task.id],
                "tags": list(task.tags),
                "partition": task.partition,
                "source_index": index,
                "task_definition_digest": task_definition_digest,
            }
        )
    return rows, list(resources.values())


def _authored_manifest(
    draft: TaskSuiteDraftV1,
    public_rows: Sequence[Mapping[str, Any]],
    *,
    task_definition_digest: str,
    criteria_digest: str,
    source_path: Path,
    source_sha256: str,
    harnesses: Sequence[str],
) -> dict[str, Any]:
    unknown = sorted(set(harnesses) - set(_HARNESS_AGENTS))
    if unknown:
        raise ValueError(
            "cannot compile authored tasks for harness(es): " + ", ".join(unknown)
        )
    return {
        "dataset": {
            "path": (TASK_DATASET_CACHE_ROOT / task_definition_digest).as_posix(),
            "materializer": "fugue.bench.task_authoring:AuthoredTaskMaterializer",
            "source": {"path": source_path.as_posix(), "sha256": source_sha256},
        },
        "k": 1,
        "n_concurrent": 1,
        "jobs_dir": f"jobs/authored-{draft.id}",
        "harnesses": [
            {"name": name, "agent": _HARNESS_AGENTS[name]} for name in harnesses
        ],
        "tasks": [
            {
                "id": str(row["id"]),
                "notes": str(row["instruction"])[:500],
                "metadata": {
                    "source_index": index,
                    "task_authoring": {
                        "task_definition_digest": task_definition_digest,
                        "criteria_digest": criteria_digest,
                        "scenario_id": row["scenario_id"],
                        "interaction": row["interaction"],
                        "environment_profile_id": row["environment"]["profile_id"],
                        "environment_kind": row["environment"]["kind"],
                        "profile_digests": row["profile_digests"],
                        "harness_applicability": row["harness_applicability"],
                        "partition": row["partition"],
                        "tags": row["tags"],
                    },
                },
                **(
                    {"repository": row["environment"]["repository"]}
                    if row["environment"].get("repository")
                    else {}
                ),
            }
            for index, row in enumerate(public_rows)
        ],
    }


def _write_authored_harbor_task(
    root: Path, case: Mapping[str, Any], repo_root: Path
) -> None:
    root.mkdir(parents=True)
    for name in ("environment", "solution", "tests"):
        (root / name).mkdir()
    environment = _mapping(case.get("environment"), "authored task environment")
    docker_lines = [f"FROM {environment['base_image']}"]
    kind = str(environment["kind"])
    repository = environment.get("repository")
    if kind == "repository":
        repo = _mapping(repository, "authored task repository")
        docker_lines.extend(
            [
                "RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git ripgrep && rm -rf /var/lib/apt/lists/*",
                "WORKDIR /workspace/repo",
                f"RUN git clone {repo['url']} . && git checkout --detach {repo['commit']} && rm -rf .git",
            ]
        )
    else:
        docker_lines.append("WORKDIR /workspace")
    for index, attachment in enumerate(case.get("attachments") or []):
        value = _mapping(attachment, "task attachment")
        relative = _safe_relative_path(value.get("locked_relative"), "locked resource")
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"locked task resource not found: {relative}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != value.get("sha256"):
            raise ValueError(f"locked task resource checksum changed: {relative}")
        local = Path("resources") / f"{index:03d}-{source.name}"
        target = root / "environment" / local
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        docker_lines.append(f"COPY {local.as_posix()} {value['target']}")
    interaction = _mapping(case.get("interaction"), "task interaction")
    (root / "environment" / "fugue-task-interaction.json").write_text(
        json.dumps(interaction, indent=2, sort_keys=True) + "\n"
    )
    docker_lines.append(
        "COPY fugue-task-interaction.json /opt/fugue/task-interaction.json"
    )
    (root / "environment" / "Dockerfile").write_text("\n".join(docker_lines) + "\n")
    (root / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.3"',
                "",
                "[task]",
                f'name = "fugue/{case["id"]}"',
                'description = "Locked Fugue authored task"',
                "",
                "[agent]",
                f"timeout_sec = {int(interaction.get('timeout_sec') or 900)}.0",
                "",
                "[verifier]",
                "timeout_sec = 60.0",
                "",
                "[environment]",
                "build_timeout_sec = 1800.0",
                f"cpus = {float(environment.get('cpus') or 2)}",
                f"memory_mb = {int(environment.get('memory_mb') or 4096)}",
                f"storage_mb = {int(environment.get('storage_mb') or 10240)}",
                "",
            ]
        )
    )
    (root / "instruction.md").write_text(
        f"# {case['title']}\n\n{case['instruction']}\n\n"
        "Write the final response to `/logs/artifacts/fugue-answer.md`. "
        "Place any requested files under `/logs/artifacts`.\n"
    )
    (root / "solution" / "solve.sh").write_text(
        "#!/bin/sh\nmkdir -p /logs/artifacts\n"
        "printf '%s\\n' 'Reference execution is intentionally unavailable.' "
        "> /logs/artifacts/fugue-answer.md\n"
    )
    (root / "tests" / "test.sh").write_text(
        "#!/bin/sh\nmkdir -p /logs/verifier\npython - <<'PY'\n"
        "import json\nfrom pathlib import Path\n"
        "answer=Path('/logs/artifacts/fugue-answer.md')\n"
        "ok=answer.is_file() and bool(answer.read_text().strip())\n"
        "Path('/logs/verifier/reward.json').write_text(json.dumps({'format_completion': float(ok)}))\n"
        "raise SystemExit(0 if ok else 1)\nPY\n"
    )
    for path in (root / "solution" / "solve.sh", root / "tests" / "test.sh"):
        path.chmod(0o755)


def _evaluate_criterion(
    criterion: CriterionV1,
    *,
    row: Mapping[str, Any],
    revision: TaskScoringRevisionV1,
    profiles: TaskProfileCatalogV1,
    policy_evidence: set[str],
    repo_root: Path,
    env: Mapping[str, str],
    judge_request: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None,
    inline_runner: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    evaluator = criterion.evaluator
    evidence = _criterion_evidence(row, revision.evidence_view, policy_evidence)
    base = {
        "criterion_id": criterion.id,
        "evaluator_type": evaluator.type,
        "required": criterion.required,
        "weight": criterion.weight,
        "threshold": criterion.threshold,
        "profile_id": evaluator.profile_id,
        "status": "unavailable",
        "score": None,
        "passed": None,
        "reason": "required evidence is unavailable",
        "cost_usd": None,
    }
    try:
        score, reason, cost = _criterion_score(
            evaluator,
            evidence=evidence,
            profiles=profiles,
            repo_root=repo_root,
            env=env,
            judge_request=judge_request,
            inline_runner=inline_runner,
        )
    except Exception as exc:
        return {**base, "reason": f"{type(exc).__name__}: {exc}"}
    if score is None:
        return {**base, "reason": reason}
    return {
        **base,
        "status": "scored",
        "score": score,
        "passed": score >= criterion.threshold,
        "reason": reason[:1000],
        "cost_usd": cost,
    }


def _criterion_score(
    evaluator: CriterionEvaluatorV1,
    *,
    evidence: Mapping[str, Any],
    profiles: TaskProfileCatalogV1,
    repo_root: Path,
    env: Mapping[str, str],
    judge_request: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None,
    inline_runner: Callable[..., dict[str, Any]] | None,
) -> tuple[float | None, str, float | None]:
    kind = evaluator.type
    config = evaluator.config
    if kind == "benchmark_outcome":
        value = evidence.get("benchmark_pass")
        return (
            (float(value), "deterministic benchmark outcome", None)
            if isinstance(value, bool)
            else (None, "benchmark outcome unavailable", None)
        )
    answer = evidence.get("answer")
    if kind == "answer_contains":
        if not isinstance(answer, str):
            return None, "answer unavailable", None
        values = [str(value).casefold() for value in config["values"]]
        matched = [value in answer.casefold() for value in values]
        return (
            sum(matched) / len(matched),
            f"matched {sum(matched)}/{len(matched)} required values",
            None,
        )
    if kind == "answer_regex":
        if not isinstance(answer, str):
            return None, "answer unavailable", None
        matched = bool(re.search(str(config["pattern"]), answer, flags=re.MULTILINE))
        return (
            float(matched),
            "regular expression matched"
            if matched
            else "regular expression did not match",
            None,
        )
    if kind == "artifact":
        observed = set(str(value) for value in evidence.get("artifact_paths") or [])
        required = set(str(value) for value in config["paths"])
        return (
            len(observed & required) / len(required),
            f"observed {len(observed & required)}/{len(required)} required artifacts",
            None,
        )
    if kind == "tool_evidence":
        observed = _observed_tools(evidence.get("tool_calls"))
        required = set(str(value) for value in config["tools"])
        if not observed:
            return None, "tool telemetry unavailable", None
        return (
            len(observed & required) / len(required),
            f"observed {len(observed & required)}/{len(required)} required tools",
            None,
        )
    if kind == "repository_diff":
        observed = set(str(value) for value in evidence.get("changed_paths") or [])
        expected = set(str(value) for value in config["paths"])
        if not observed:
            return 0.0, "no changed paths observed", None
        return (
            len(observed & expected) / len(expected),
            f"changed {len(observed & expected)}/{len(expected)} relevant paths",
            None,
        )
    if kind == "judge":
        profile = profiles.judge(str(evaluator.profile_id))
        request = judge_request or _judge_request
        payload, usage = request(profile=profile, evidence=evidence, env=env)
        score = _unit_number(payload.get("score"), "judge score")
        reason = _bounded_text(payload.get("reason"), "judge reason", 1000)
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cost = (
            input_tokens * profile.input_cost_per_million
            + output_tokens * profile.output_cost_per_million
        ) / 1_000_000
        return score, reason, cost
    if kind == "inline_python":
        profile = profiles.scorer_runtime(str(evaluator.runtime_profile_id))
        runner = inline_runner or run_inline_scorer
        payload = runner(
            source=str(evaluator.source),
            evidence=evidence,
            reference=evaluator.config,
            profile=profile,
            limits=_inline_limits(evaluator.config),
        )
        return (
            float(payload["score"]),
            str(payload.get("reason") or "inline scorer"),
            0.0,
        )
    raise AssertionError(f"unhandled evaluator: {kind}")


def _criterion_evidence(
    row: Mapping[str, Any], evidence_view: str, allowed: set[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "answer" in allowed:
        result["answer"] = row.get("agent_response") or row.get("answer")
    if "benchmark" in allowed:
        result["benchmark_status"] = row.get("status")
        result["benchmark_pass"] = row.get("pass")
    if evidence_view == "answer_artifacts_tools":
        if "artifacts" in allowed:
            result["artifact_paths"] = row.get("artifact_paths") or row.get("artifacts")
        if "tool_calls" in allowed:
            result["tool_calls"] = row.get("weave_tool_names") or row.get("tool_names")
        if "changed_paths" in allowed:
            result["changed_paths"] = row.get("changed_paths")
        if "opened_paths" in allowed:
            result["opened_paths"] = row.get("opened_paths")
        if "trace_summary" in allowed:
            result["trace_summary"] = row.get("trace_summary")
    return result


def _criteria_outcome(
    criterion_set: CriteriaSetV1, results: Sequence[Mapping[str, Any]]
) -> tuple[str, float | None, bool | None]:
    required_unavailable = any(
        bool(result["required"]) and result["status"] != "scored" for result in results
    )
    if required_unavailable:
        return "unavailable", None, None
    applicable = [result for result in results if result["status"] == "scored"]
    if not applicable:
        return "unavailable", None, None
    total_weight = sum(float(result["weight"]) for result in applicable)
    score = (
        sum(float(result["score"]) * float(result["weight"]) for result in applicable)
        / total_weight
    )
    required_pass = all(
        not bool(result["required"]) or bool(result["passed"]) for result in applicable
    )
    passed = required_pass and score >= criterion_set.pass_threshold
    return "scored", score, passed


def _judge_request(
    *, profile: JudgeProfileV1, evidence: Mapping[str, Any], env: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from fugue.bench.evaluations import _post_judge
    from fugue.model_plane import resolve_model_route

    route = resolve_model_route(profile.model, env)
    api_key = str(env.get(route.api_key_env) or "").strip()
    if not api_key:
        raise RuntimeError(f"{route.api_key_env} is required for task judging")
    prompt = (
        profile.prompt
        + "\nReturn only JSON with score (0..1) and reason.\n"
        + json.dumps({"evidence": evidence}, sort_keys=True, default=str)[:48_000]
    )
    import httpx

    with httpx.Client(timeout=120) as client:
        payload, usage = _post_judge(client, route, api_key, env, prompt)
    if not isinstance(payload, dict):
        raise ValueError("judge must return one JSON object")
    return payload, usage


def _group_results(
    rows: Sequence[Mapping[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key) or "") for key in keys)].append(row)
    result: list[dict[str, Any]] = []
    for group_key, values in sorted(groups.items()):
        scored = [item for item in values if item.get("criteria_pass") is not None]
        result.append(
            {
                **dict(zip(keys, group_key, strict=True)),
                "predictions": len(values),
                "scored": len(scored),
                "criteria_passes": sum(
                    bool(item.get("criteria_pass")) for item in scored
                ),
                "criteria_pass_rate": (
                    sum(bool(item.get("criteria_pass")) for item in scored)
                    / len(scored)
                    if scored
                    else None
                ),
                "benchmark_passes": sum(
                    item.get("benchmark_pass") is True for item in values
                ),
                "unavailable": len(values) - len(scored),
            }
        )
    return result


def _scenario_results(
    rows: Sequence[Mapping[str, Any]], scenarios: Mapping[str, TaskScenarioV1]
) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario_id"])].append(row)
    result: list[dict[str, Any]] = []
    for scenario_id, values in sorted(by_scenario.items()):
        scenario = scenarios[scenario_id]
        weights = {item.task_id: item.weight for item in scenario.tasks}
        must_pass = {item.task_id for item in scenario.tasks if item.must_pass}
        task_means: dict[str, float] = {}
        for task_id in weights:
            scored = [
                row
                for row in values
                if row["task_id"] == task_id and row.get("criteria_score") is not None
            ]
            if scored:
                task_means[task_id] = sum(
                    float(row["criteria_score"]) for row in scored
                ) / len(scored)
        denominator = sum(weights[task] for task in task_means)
        score = (
            sum(task_means[task] * weights[task] for task in task_means) / denominator
            if denominator
            else None
        )
        required_pass = all(
            any(
                row["task_id"] == task and row.get("criteria_pass") is True
                for row in values
            )
            for task in must_pass
        )
        result.append(
            {
                "scenario_id": scenario_id,
                "tasks": len(scenario.tasks),
                "score": score,
                "required_tasks_passed": required_pass,
            }
        )
    return result


def _aligned_contrasts(
    rows: Sequence[Mapping[str, Any]], samples: int, seed_value: str
) -> list[dict[str, Any]]:
    harnesses = sorted({str(row["harness"]) for row in rows})
    return [
        _contrast(rows, left, right, samples, f"{seed_value}:{left}:{right}")
        for index, left in enumerate(harnesses)
        for right in harnesses[index + 1 :]
    ]


def _contrast(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    samples: int,
    seed_value: str,
) -> dict[str, Any]:
    paired: dict[str, list[float]] = defaultdict(list)
    coordinates: dict[tuple[str, int, str], bool] = {}
    for row in rows:
        if row.get("criteria_pass") is None:
            continue
        coordinates[
            (str(row["task_id"]), int(row["trial_index"]), str(row["harness"]))
        ] = bool(row["criteria_pass"])
    tasks = sorted({task for task, _, _ in coordinates})
    for task in tasks:
        trials = sorted({trial for item, trial, _ in coordinates if item == task})
        for trial in trials:
            if (task, trial, left) in coordinates and (
                task,
                trial,
                right,
            ) in coordinates:
                paired[task].append(
                    float(coordinates[(task, trial, left)])
                    - float(coordinates[(task, trial, right)])
                )
    observed = _cluster_mean(paired)
    low, high = _cluster_bootstrap(paired, samples, seed_value)
    return {
        "left_harness": left,
        "right_harness": right,
        "aligned_pairs": sum(len(value) for value in paired.values()),
        "task_clusters": len(paired),
        "pass_rate_difference": observed,
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
    }


def _scenario_interactions(
    rows: Sequence[Mapping[str, Any]], samples: int, seed_value: str
) -> list[dict[str, Any]]:
    scenarios = sorted({str(row["scenario_id"]) for row in rows})
    harnesses = sorted({str(row["harness"]) for row in rows})
    result: list[dict[str, Any]] = []
    for scenario in scenarios:
        selected = [row for row in rows if row["scenario_id"] == scenario]
        for index, left in enumerate(harnesses):
            for right in harnesses[index + 1 :]:
                value = _contrast(
                    selected,
                    left,
                    right,
                    samples,
                    f"{seed_value}:{scenario}:{left}:{right}",
                )
                result.append({"scenario_id": scenario, **value})
    return result


def _judge_sensitivity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for criterion in row.get("criteria") or []:
            if (
                criterion.get("evaluator_type") != "judge"
                or criterion.get("score") is None
            ):
                continue
            groups[str(criterion.get("profile_id") or "unknown")].append(
                float(criterion["score"])
            )
    return [
        {
            "judge_profile_id": profile,
            "scores": len(values),
            "mean_score": sum(values) / len(values),
        }
        for profile, values in sorted(groups.items())
    ]


def _cluster_mean(values: Mapping[str, Sequence[float]]) -> float | None:
    means = [sum(items) / len(items) for items in values.values() if items]
    return sum(means) / len(means) if means else None


def _cluster_bootstrap(
    values: Mapping[str, Sequence[float]], samples: int, seed_value: str
) -> tuple[float | None, float | None]:
    tasks = sorted(values)
    if not tasks:
        return None, None
    rng = random.Random(int(hashlib.sha256(seed_value.encode()).hexdigest()[:16], 16))
    estimates: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(tasks) for _ in tasks]
        means = [sum(values[task]) / len(values[task]) for task in selected]
        estimates.append(sum(means) / len(means))
    estimates.sort()
    low_index = max(0, math.floor(0.025 * (len(estimates) - 1)))
    high_index = min(len(estimates) - 1, math.ceil(0.975 * (len(estimates) - 1)))
    return estimates[low_index], estimates[high_index]


def _task_scenario_map(raw: Sequence[Any]) -> dict[str, str]:
    return {
        item.task_id: scenario.id
        for scenario in (_scenario(value) for value in raw)
        for item in scenario.tasks
    }


def _validate_evaluator_config(
    kind: str,
    profile_id: str | None,
    runtime_id: str | None,
    source: str | None,
    config: Mapping[str, Any],
) -> None:
    expected: dict[str, set[str]] = {
        "benchmark_outcome": set(),
        "answer_contains": {"values"},
        "answer_regex": {"pattern"},
        "artifact": {"paths"},
        "tool_evidence": {"tools"},
        "repository_diff": {"paths"},
        "judge": set(),
        "inline_python": set(config),
    }
    if kind != "inline_python":
        _reject_unknown(config, expected[kind], f"{kind} evaluator config")
        if set(config) != expected[kind]:
            missing = sorted(expected[kind] - set(config))
            raise ValueError(f"{kind} evaluator config missing: {', '.join(missing)}")
    if kind in {"answer_contains", "artifact", "tool_evidence", "repository_diff"}:
        key = next(iter(expected[kind]))
        values = config.get(key)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise ValueError(f"{kind} evaluator {key} must be a non-empty string list")
    if kind == "answer_regex":
        re.compile(str(config.get("pattern") or ""))
    if kind == "judge" and (not profile_id or runtime_id or source):
        raise ValueError("judge evaluators require only profile_id")
    if kind == "inline_python":
        if not runtime_id or not source or profile_id:
            raise ValueError(
                "inline_python evaluators require runtime_profile_id and source"
            )
        if len(source) > _MAX_CODE:
            raise ValueError(f"inline scorer source exceeds {_MAX_CODE} characters")
    if kind not in {"judge", "inline_python"} and (profile_id or runtime_id or source):
        raise ValueError(f"{kind} evaluator does not accept profiles or source")


def _inline_limits(config: Mapping[str, Any]) -> TaskAuthoringLimitsV1:
    raw = config.get("limits")
    if not isinstance(raw, dict):
        raise ValueError("inline scorer config requires locked limits")
    return TaskAuthoringLimitsV1(
        max_tasks=1,
        max_scenarios=1,
        max_prompt_bytes=1,
        max_authored_asset_bytes=1,
        max_user_turns=1,
        max_agent_turns=1,
        max_interactor_calls=0,
        max_judge_calls=0,
        scorer_timeout_sec=_positive_int(raw.get("timeout_sec"), "inline timeout"),
        scorer_memory_mb=_positive_int(raw.get("memory_mb"), "inline memory"),
        scorer_cpus=_positive_number(raw.get("cpus"), "inline cpus"),
        scorer_output_bytes=_positive_int(raw.get("output_bytes"), "inline output"),
    )


def _scorer_payload(payload: Mapping[str, Any]) -> None:
    _reject_unknown(payload, {"score", "reason", "details"}, "inline scorer output")
    _unit_number(payload.get("score"), "inline scorer score")
    if payload.get("reason") is not None:
        _bounded_text(payload["reason"], "inline scorer reason", 1000)
    if payload.get("details") is not None and not isinstance(payload["details"], dict):
        raise ValueError("inline scorer details must be an object")


def _observed_tools(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key, count in value.items() if count}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def _allowed_profile(
    profile_id: str,
    allowed: Sequence[str],
    getter: Callable[[str], Any],
    label: str,
    failures: list[str],
) -> Any | None:
    if profile_id not in allowed:
        failures.append(f"{label} profile {profile_id} is not allowed")
        return None
    try:
        return getter(profile_id)
    except KeyError:
        failures.append(f"{label} profile {profile_id} is not registered")
        return None


def _with_profile_digest(profile: Any, raw: Mapping[str, Any]) -> Any:
    digest = _artifact_digest(profile.to_dict(), "profile_digest")
    supplied = str(raw.get("profile_digest") or "")
    if supplied and supplied != digest:
        raise ValueError(f"profile digest does not match {profile.id}")
    return replace(profile, profile_digest=digest)


def _profile(values: Sequence[Any], profile_id: str, label: str) -> Any:
    match = next((item for item in values if item.id == profile_id), None)
    if match is None:
        raise KeyError(f"unknown {label} profile: {profile_id}")
    return match


def _artifact_digest(value: Mapping[str, Any], field: str) -> str:
    return stable_digest({**value, field: ""})


def _verify_artifact(value: Mapping[str, Any], field: str, label: str) -> None:
    supplied = _required_digest(value.get(field), field)
    if supplied != _artifact_digest(value, field):
        raise ValueError(f"{field} does not match the {label}")


def _schema(raw: Mapping[str, Any], label: str) -> int:
    if int(raw.get("schema_version") or 0) != TASK_AUTHORING_SCHEMA_VERSION:
        raise ValueError(f"{label} must use schema_version 1")
    return TASK_AUTHORING_SCHEMA_VERSION


def _reject_unknown(raw: Mapping[str, Any], known: set[str], label: str) -> None:
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    return list(value)


def _mapping_tuple(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    return tuple(_mapping(item, label.removesuffix("s")) for item in value)


def _text_tuple(
    value: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if value in (None, []) and allow_empty:
        return ()
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        raise ValueError(
            f"{label} must be a {'possibly empty ' if allow_empty else 'non-empty '}list"
        )
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{label} values must be non-empty")
    _require_unique(result, label)
    return result


def _bounded_text_tuple(
    value: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    result = _text_tuple(value, label, allow_empty=allow_empty)
    return tuple(_bounded_text(item, label, 4000) for item in result)


def _id_tuple(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    return tuple(
        validate_id(item, kind=f"{label} id")
        for item in _text_tuple(value, label, allow_empty=allow_empty)
    )


def _harness_tuple(value: Any) -> tuple[str, ...]:
    harnesses = _id_tuple(value, "harness")
    unknown = sorted(set(harnesses) - set(_SAFE_HARNESSES))
    if unknown:
        raise ValueError("unknown authored-task harness(es): " + ", ".join(unknown))
    return harnesses


def _bounded_text(value: Any, label: str, limit: int) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} is required")
    if len(result) > limit:
        raise ValueError(f"{label} exceeds {limit} characters")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _positive_number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _non_negative_number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be non-negative and finite")
    return result


def _unit_number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def _required_digest(value: Any, label: str) -> str:
    result = str(value or "")
    if not _DIGEST.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _digest_mapping(value: Any, label: str) -> dict[str, str]:
    raw = _mapping(value, label)
    return {
        str(key): _required_digest(item, f"{label} {key}") for key, item in raw.items()
    }


def _safe_relative_path(value: Any, label: str) -> str:
    path = Path(str(value or ""))
    if not str(path) or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path")
    return path.as_posix()


def _safe_container_target(value: Any) -> str:
    path = PurePosixPath(str(value or ""))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("resource target must be an absolute container path")
    if not path.is_relative_to(PurePosixPath("/workspace/resources")):
        raise ValueError("resource target must stay under /workspace/resources")
    return path.as_posix()


def _require_unique(values: Sequence[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}(s): {', '.join(duplicates)}")


def _drop_empty(
    value: dict[str, Any], *, preserve_false: bool = False
) -> dict[str, Any]:
    empty = (None, "", [], {}, ()) if preserve_false else (None, "", [], {}, (), False)
    return {key: item for key, item in value.items() if item not in empty}


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))
