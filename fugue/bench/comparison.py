from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.library import ExperimentSpec, experiment_from_data, validate_id
from fugue.bench.operator import (
    ExperimentRequest,
    OperatorService,
    PreviewSummary,
)
from fugue.research.approvals import ApprovalLedger
from fugue.research.store import StudyStore

COMPARISON_SCHEMA_VERSION = 1
COMPARISON_RUNTIME_ROOT = Path(".fugue/runtime/comparisons")
COMPARISON_PRIVATE_ROOT = Path(".fugue/private/comparisons")
COMPARISON_RESULT_ROOT = Path(".fugue/results/comparisons")

_HARNESS_AGENTS = {
    "hermes": "fugue.agents:FugueHermes",
    "openclaw": "fugue.agents:FugueOpenClaw",
    "claude-code": "fugue.agents:FugueClaudeCode",
    "codex": "fugue.agents:FugueCodex",
}
_READINESS = frozenset(
    {"ready", "needs_review", "blocked", "no_comparison_justified"}
)
_PUBLIC_TASK_FIELDS = frozenset({"id", "input", "resources", "tags", "partition"})
_PRIVATE_LABEL_FIELDS = frozenset(
    {"id", "expected", "base_output", "gold_output"}
)
_PRIVATE_WORDS = frozenset(
    {"expected", "gold", "reference_answer", "private", "answer_key"}
)
_COMPARISON_BASE_IMAGE = (
    "python:3.12.10-slim-bookworm@"
    "sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db"
)


@dataclass(frozen=True)
class ComparisonTasksetV1:
    tasks: str
    private_labels: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonCandidateV1:
    label: str
    prompt_id: str | None = None
    skills: tuple[str, ...] = ()
    context: dict[str, Any] = field(
        default_factory=lambda: {"system_id": "none", "delivery": "portable"}
    )
    integrations: tuple[dict[str, Any], ...] = ()
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))

    def behavior(self) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("label", None)
        return value


@dataclass(frozen=True)
class ComparisonEvaluatorV1:
    id: str
    type: Literal["deterministic", "llm_judge"]
    required: bool
    checks: tuple[str, ...] = ()
    scorer: str | None = None
    runtime: str | None = None
    profile: str | None = None
    calibration: str | None = None
    rubric: str | None = None
    dimensions: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    reserve_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ComparisonExecutionPolicyV1:
    model: str
    harnesses: tuple[str, ...]
    attempts: int
    concurrency: int
    max_cost_usd: float
    reserve_per_attempt_usd: float
    approval_required: bool
    trace_content: Literal["full", "metadata"]
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self), preserve_false=True)


@dataclass(frozen=True)
class ComparisonSpecV1:
    schema_version: int
    id: str
    question: str
    taskset: ComparisonTasksetV1
    baseline: ComparisonCandidateV1
    candidate: ComparisonCandidateV1
    changed: tuple[str, ...]
    evaluators: tuple[ComparisonEvaluatorV1, ...]
    execution: ComparisonExecutionPolicyV1
    spec_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        execution = value.get("execution")
        if isinstance(execution, dict) and not execution.get("environment"):
            execution.pop("environment", None)
        return value


@dataclass(frozen=True)
class ComparisonReadinessV1:
    schema_version: int
    comparison_id: str
    question: str
    task_count: int
    taskset_digest: str
    private_labels_digest: str
    actual_changes: tuple[str, ...]
    declared_changes: tuple[str, ...]
    base_failures: int
    gold_passes: int
    deterministic_evaluators: tuple[str, ...]
    judge_evaluators: tuple[str, ...]
    evaluator_digests: dict[str, str]
    attempts: int
    estimated_cells: int
    estimated_cost_usd: float
    status: Literal[
        "ready", "needs_review", "blocked", "no_comparison_justified"
    ]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    readiness_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonPreviewV1:
    schema_version: int
    comparison: dict[str, Any]
    readiness: dict[str, Any]
    matrix: dict[str, Any]
    experiment: dict[str, Any]
    manifest: dict[str, Any]
    preview_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonResultV1:
    schema_version: int
    comparison_id: str
    preview_digest: str
    source: str
    rows: int
    baseline_passed: int
    candidate_passed: int
    improved: int
    regressed: int
    unchanged: int
    incomplete: int
    required_evaluations_incomplete: int
    deterministic_summary: dict[str, Any]
    judge_summary: dict[str, Any]
    mechanism_summary: dict[str, Any]
    operational_summary: dict[str, Any]
    evidence_links: tuple[dict[str, str], ...]
    paired_cases: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    result_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_comparison(path: Path, *, repo_root: Path) -> ComparisonSpecV1:
    resolved = _safe_input_path(path, repo_root, "comparison")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("comparison YAML must be a mapping")
    return comparison_from_dict(raw, repo_root=repo_root, source=resolved.parent)


def comparison_from_dict(
    raw: Mapping[str, Any], *, repo_root: Path, source: Path | None = None
) -> ComparisonSpecV1:
    _reject_unknown(
        raw,
        {
            "schema_version",
            "id",
            "question",
            "taskset",
            "baseline",
            "candidate",
            "changed",
            "evaluators",
            "execution",
            "spec_digest",
        },
        "comparison",
    )
    version = _schema(raw, "comparison")
    base = source or repo_root
    taskset_raw = _mapping(raw.get("taskset"), "taskset")
    _reject_unknown(taskset_raw, {"tasks", "private_labels"}, "taskset")
    taskset = ComparisonTasksetV1(
        tasks=_portable_input_path(taskset_raw.get("tasks"), base, repo_root, "tasks"),
        private_labels=_portable_input_path(
            taskset_raw.get("private_labels"),
            base,
            repo_root,
            "private labels",
        ),
    )
    parsed_evaluators = tuple(
        _evaluator(item)
        for item in _sequence(raw.get("evaluators"), "evaluators")
    )
    evaluators = tuple(
        replace(
            evaluator,
            scorer=(
                _portable_input_path(
                    evaluator.scorer,
                    base,
                    repo_root,
                    "deterministic scorer",
                )
                if evaluator.scorer
                else None
            ),
            calibration=(
                _portable_input_path(
                    evaluator.calibration,
                    base,
                    repo_root,
                    "judge calibration",
                )
                if evaluator.calibration
                else None
            ),
        )
        for evaluator in parsed_evaluators
    )
    if len({item.id for item in evaluators}) != len(evaluators):
        raise ValueError("comparison evaluator ids must be unique")
    execution = _execution(raw.get("execution"))
    changed = _string_tuple(raw.get("changed"), "changed dimension")
    if len(set(changed)) != len(changed):
        raise ValueError("declared changed dimensions must be unique")
    unsigned = ComparisonSpecV1(
        schema_version=version,
        id=validate_id(str(raw.get("id") or ""), kind="comparison id"),
        question=_text(raw.get("question"), "comparison question", 1000),
        taskset=taskset,
        baseline=_candidate(raw.get("baseline"), "Baseline"),
        candidate=_candidate(raw.get("candidate"), "Candidate"),
        changed=changed,
        evaluators=evaluators,
        execution=execution,
    )
    digest = _artifact_digest(unsigned.to_dict(), "spec_digest")
    supplied = str(raw.get("spec_digest") or "")
    if supplied and supplied != digest:
        raise ValueError("comparison spec digest does not match")
    return replace(unsigned, spec_digest=digest)


def check_comparison(
    spec: ComparisonSpecV1, *, repo_root: Path
) -> ComparisonReadinessV1:
    from fugue.bench.integrations import load_integration
    from fugue.bench.sources import resolve_skill

    tasks = _load_public_tasks(repo_root / spec.taskset.tasks)
    labels = _load_private_labels(repo_root / spec.taskset.private_labels)
    actual_changes, blockers = _comparison_identity_issues(spec)
    warnings: list[str] = []
    task_ids = tuple(str(item["id"]) for item in tasks)
    blockers.extend(_task_label_issues(task_ids, labels))
    for candidate_name, candidate in (
        ("baseline", spec.baseline),
        ("candidate", spec.candidate),
    ):
        for skill_id in candidate.skills:
            try:
                resolve_skill(skill_id, repo_root)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                blockers.append(
                    f"{candidate_name} Skill {skill_id!r} is not locked and usable: {exc}"
                )
        for selection in candidate.integrations:
            integration_id = str(selection["id"])
            try:
                integration = load_integration(integration_id, repo_root)
            except (FileNotFoundError, ValueError) as exc:
                blockers.append(
                    f"{candidate_name} integration {integration_id!r} is not locked "
                    f"and usable: {exc}"
                )
                continue
            if integration.support != "supported":
                warnings.append(
                    f"{candidate_name} integration {integration_id!r} is "
                    f"{integration.support}; its evidence is exploratory"
                )
    deterministic = tuple(
        item.id for item in spec.evaluators if item.type == "deterministic"
    )
    judges = tuple(item.id for item in spec.evaluators if item.type == "llm_judge")
    if not deterministic:
        blockers.append("at least one deterministic evaluator is required")
    required_checks = {
        check for item in spec.evaluators if item.type == "deterministic" for check in item.checks
    }
    unsupported_checks = sorted(
        required_checks - {"answer_present", "expected_values"}
    )
    if unsupported_checks:
        blockers.append(
            "unsupported deterministic checks: " + ", ".join(unsupported_checks)
        )
    (
        base_failures,
        gold_passes,
        qualification_blockers,
        qualification_warnings,
    ) = (
        _qualification_results(
            tasks,
            labels,
            tuple(
                item
                for item in spec.evaluators
                if item.type == "deterministic"
            ),
            repo_root=repo_root,
        )
    )
    blockers.extend(qualification_blockers)
    warnings.extend(qualification_warnings)
    labels_by_id = {str(item["id"]): item for item in labels}
    if tasks and base_failures == 0 and all(
        "base_output" in labels_by_id.get(task_id, {}) for task_id in task_ids
    ):
        warnings.append("the baseline fixtures pass every task; the cohort is saturated")
    if spec.execution.attempts < 2:
        warnings.append(
            "one attempt cannot estimate ordinary run-to-run variation"
        )
    for judge in (item for item in spec.evaluators if item.type == "llm_judge"):
        issue = _judge_calibration_issue(judge, repo_root)
        if issue:
            (blockers if judge.required else warnings).append(issue)
    estimated_cells = (
        len(tasks)
        * 2
        * len(spec.execution.harnesses)
        * spec.execution.attempts
    )
    judge_reserve = sum(
        item.reserve_cost_usd
        for item in spec.evaluators
        if item.type == "llm_judge"
    )
    estimated_cost = estimated_cells * (
        spec.execution.reserve_per_attempt_usd + judge_reserve
    )
    if estimated_cells < 1:
        blockers.append("comparison must resolve at least one attempt")
    if estimated_cost > spec.execution.max_cost_usd + 1e-9:
        blockers.append(
            f"estimated cost ${estimated_cost:.2f} exceeds the "
            f"${spec.execution.max_cost_usd:.2f} comparison limit"
        )
    if blockers:
        status = "blocked"
    elif tasks and base_failures == 0:
        status = "no_comparison_justified"
    elif warnings:
        status = "needs_review"
    else:
        status = "ready"
    unsigned = ComparisonReadinessV1(
        schema_version=COMPARISON_SCHEMA_VERSION,
        comparison_id=spec.id,
        question=spec.question,
        task_count=len(tasks),
        taskset_digest=_sha256_path(repo_root / spec.taskset.tasks),
        private_labels_digest=_sha256_path(
            repo_root / spec.taskset.private_labels
        ),
        actual_changes=actual_changes,
        declared_changes=spec.changed,
        base_failures=base_failures,
        gold_passes=gold_passes,
        deterministic_evaluators=deterministic,
        judge_evaluators=judges,
        evaluator_digests={
            item.id: _evaluator_digest(item, repo_root)
            for item in spec.evaluators
        },
        attempts=spec.execution.attempts,
        estimated_cells=estimated_cells,
        estimated_cost_usd=round(estimated_cost, 6),
        status=status,  # type: ignore[arg-type]
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
    return replace(
        unsigned,
        readiness_digest=_artifact_digest(
            unsigned.to_dict(), "readiness_digest"
        ),
    )


def _comparison_identity_issues(
    spec: ComparisonSpecV1,
) -> tuple[tuple[str, ...], list[str]]:
    actual = tuple(
        sorted(_behavior_diff(spec.baseline.behavior(), spec.candidate.behavior()))
    )
    blockers: list[str] = []
    if not actual:
        blockers.append("baseline and candidate have identical behavior")
    if set(actual) != set(spec.changed):
        blockers.append(
            "declared candidate changes do not match the resolved behavior diff "
            f"(declared={list(spec.changed)}, actual={list(actual)})"
        )
    return actual, blockers


def _task_label_issues(
    task_ids: Sequence[str], labels: Sequence[Mapping[str, Any]]
) -> list[str]:
    label_ids = {str(item["id"]) for item in labels}
    task_set = set(task_ids)
    issues: list[str] = []
    missing = sorted(task_set - label_ids)
    extra = sorted(label_ids - task_set)
    if missing:
        issues.append("private labels are missing tasks: " + ", ".join(missing))
    if extra:
        issues.append("private labels reference unknown tasks: " + ", ".join(extra))
    return issues


def _qualification_results(
    tasks: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    evaluators: Sequence[ComparisonEvaluatorV1],
    *,
    repo_root: Path,
) -> tuple[int, int, list[str], list[str]]:
    values = {str(item["id"]): item for item in labels}
    blockers: list[str] = []
    warnings: list[str] = []
    base_failures = 0
    gold_passes = 0
    for task in tasks:
        task_id = str(task["id"])
        label = values.get(task_id)
        if label is None:
            continue
        if "base_output" not in label:
            warnings.append(f"{task_id}: missing base_output qualification fixture")
        else:
            try:
                base_passed, _ = _score_deterministic_output(
                    task=task,
                    output=label["base_output"],
                    expected=label.get("expected"),
                    evidence={},
                    evaluators=evaluators,
                    repo_root=repo_root,
                )
            except Exception as exc:
                blockers.append(
                    f"{task_id}: evaluator qualification failed: "
                    f"{type(exc).__name__}"
                )
            else:
                if not base_passed:
                    base_failures += 1
        if "gold_output" not in label:
            warnings.append(f"{task_id}: missing gold_output qualification fixture")
        else:
            try:
                gold_passed, _ = _score_deterministic_output(
                    task=task,
                    output=label["gold_output"],
                    expected=label.get("expected"),
                    evidence={},
                    evaluators=evaluators,
                    repo_root=repo_root,
                )
            except Exception as exc:
                blockers.append(
                    f"{task_id}: evaluator qualification failed: "
                    f"{type(exc).__name__}"
                )
            else:
                if gold_passed:
                    gold_passes += 1
                else:
                    blockers.append(
                        f"{task_id}: known-good output fails the evaluator"
                    )
    return base_failures, gold_passes, blockers, warnings


def preview_comparison(
    spec: ComparisonSpecV1,
    *,
    repo_root: Path,
    operator: OperatorService | None = None,
) -> ComparisonPreviewV1:
    readiness = check_comparison(spec, repo_root=repo_root)
    experiment, manifest, public_rows = compile_comparison(spec, repo_root=repo_root)
    manifest_path = Path(experiment.manifest)
    overlay = {
        manifest_path.as_posix(): yaml.safe_dump(manifest, sort_keys=False),
    }
    service = operator or OperatorService(repo_root)
    matrix = service.preview_experiment(
        experiment,
        request=ExperimentRequest(
            experiment_id=experiment.id,
            n_concurrent=spec.execution.concurrency,
        ),
        asset_overlay=overlay,
    )
    if matrix.estimated_trials != readiness.estimated_cells:
        raise RuntimeError(
            "comparison compiler and OperatorService resolved different attempt counts"
        )
    unsigned = ComparisonPreviewV1(
        schema_version=COMPARISON_SCHEMA_VERSION,
        comparison=spec.to_dict(),
        readiness=readiness.to_dict(),
        matrix=_preview_dict(matrix),
        experiment=experiment.to_dict(),
        manifest=manifest,
    )
    return replace(
        unsigned,
        preview_digest=_artifact_digest(unsigned.to_dict(), "preview_digest"),
    )


def compile_comparison(
    spec: ComparisonSpecV1, *, repo_root: Path
) -> tuple[ExperimentSpec, dict[str, Any], list[dict[str, Any]]]:
    tasks = _load_public_tasks(repo_root / spec.taskset.tasks)
    public_rows = [
        _public_case(task, spec=spec, index=index, repo_root=repo_root)
        for index, task in enumerate(tasks)
    ]
    public_text = _jsonl(public_rows)
    taskset_digest = hashlib.sha256(public_text.encode()).hexdigest()
    evaluator_digests = {
        item.id: _evaluator_digest(item, repo_root) for item in spec.evaluators
    }
    runtime = COMPARISON_RUNTIME_ROOT / spec.spec_digest
    source_path = runtime / "public-cases.jsonl"
    manifest_path = runtime / "manifest.yaml"
    dataset_path = Path(".fugue/cache/simple-task-datasets") / taskset_digest
    manifest = {
        "dataset": {
            "path": dataset_path.as_posix(),
            "materializer": "fugue.bench.task_authoring:AuthoredTaskMaterializer",
            "source": {
                "path": source_path.as_posix(),
                "sha256": hashlib.sha256(public_text.encode()).hexdigest(),
            },
        },
        "model": spec.execution.model,
        "k": 1,
        "n_concurrent": spec.execution.concurrency,
        "jobs_dir": f".fugue/runtime/jobs/{spec.id}",
        "harnesses": [
            {"name": name, "agent": _HARNESS_AGENTS[name]}
            for name in spec.execution.harnesses
        ],
        "tasks": [
            {
                "id": row["id"],
                "notes": row["instruction"][:500],
                "metadata": {
                    "source_index": index,
                    "task_authoring": {
                        "task_definition_digest": taskset_digest,
                        "criteria_digest": _sha256_path(
                            repo_root / spec.taskset.private_labels
                        ),
                        "scenario_id": "comparison",
                        "interaction": {
                            "type": "single_turn",
                            "max_user_turns": 0,
                            "max_agent_turns": 1,
                        },
                        "interaction_controller": row["interaction"],
                        "environment_profile_id": "artifact-python-v1",
                        "environment_kind": "artifact",
                        "profile_digests": {
                            f"comparison-evaluator:{evaluator_id}": digest
                            for evaluator_id, digest in evaluator_digests.items()
                        },
                        "harness_applicability": row["harness_applicability"],
                        "partition": row["partition"],
                        "tags": row["tags"],
                    },
                },
            }
            for index, row in enumerate(public_rows)
        ],
    }
    experiment = experiment_from_data(
        {
            "id": spec.id,
            "title": spec.question,
            "description": "Compiled Fugue Agent-change comparison.",
            "manifest": manifest_path.as_posix(),
            "model": spec.execution.model,
            "run_name": spec.id,
            "tags": ["comparison", "technical-preview"],
            "harnesses": list(spec.execution.harnesses),
            "variants": [
                _variant_dict("baseline", spec.baseline),
                _variant_dict("candidate", spec.candidate),
            ],
            "n_attempts": spec.execution.attempts,
            "n_concurrent": spec.execution.concurrency,
            "n_tasks": len(tasks),
            "jobs_dir": f".fugue/runtime/jobs/{spec.id}",
            "trace_content": spec.execution.trace_content,
            "environment": spec.execution.environment,
            "research_view": {
                "observation": spec.question,
                "rationale": "Test one declared Agent-system change on aligned tasks.",
                "success_definition": "Pass the required deterministic evaluator.",
                "task_title": f"{len(tasks)} locked comparison tasks",
                "task_summary": "Public task inputs with host-only expected values.",
                "interaction_mode": "Single turn",
                "base_instruction_summary": "Use the common task instruction.",
                "treatment_summaries": {
                    "baseline": spec.baseline.label,
                    "candidate": spec.candidate.label,
                },
                "pass_rule": "All required deterministic checks must pass.",
                "scorers": [
                    _research_scorer(
                        item,
                        revision=evaluator_digests[item.id],
                    )
                    for item in spec.evaluators
                ],
            },
        }
    )
    return experiment, manifest, public_rows


def materialize_comparison(
    preview: ComparisonPreviewV1, *, repo_root: Path
) -> tuple[ExperimentSpec, ExperimentRequest]:
    _verify_artifact(preview.to_dict(), "preview_digest", "comparison preview")
    spec = comparison_from_dict(
        preview.comparison,
        repo_root=repo_root,
        source=repo_root,
    )
    current = preview_comparison(spec, repo_root=repo_root)
    if current.preview_digest != preview.preview_digest:
        raise ValueError("comparison inputs changed after preview")
    experiment, manifest, public_rows = compile_comparison(spec, repo_root=repo_root)
    root = repo_root / COMPARISON_RUNTIME_ROOT / spec.spec_digest
    root.mkdir(parents=True, exist_ok=True)
    _atomic_text(root / "public-cases.jsonl", _jsonl(public_rows))
    _atomic_text(root / "manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))
    _atomic_text(
        root / "comparison.yaml",
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
    )
    private_root = repo_root / COMPARISON_PRIVATE_ROOT / spec.spec_digest
    private_root.mkdir(parents=True, exist_ok=True)
    private_target = private_root / "labels.jsonl"
    shutil.copyfile(repo_root / spec.taskset.private_labels, private_target)
    private_target.chmod(0o600)
    atomic_write_json(root / "preview.json", preview.to_dict())
    return experiment, ExperimentRequest(
        experiment_id=spec.id,
        n_concurrent=spec.execution.concurrency,
        n_attempts=spec.execution.attempts,
        n_tasks=int(preview.readiness["task_count"]),
    )


def claim_comparison_approval(
    preview: ComparisonPreviewV1,
    *,
    approval_digest: str,
    repo_root: Path,
) -> None:
    readiness = ComparisonReadinessV1(**preview.readiness)
    if readiness.status != "ready":
        raise ValueError(
            f"comparison is {readiness.status}; only ready comparisons may run"
        )
    store = StudyStore(repo_root)
    ApprovalLedger(store.path).claim(
        approval_digest=approval_digest,
        subject_kind="experiment",
        preview_digest=preview.preview_digest,
        subject_id=f"comparison-{preview.preview_digest[:20]}",
        estimated_cells=readiness.estimated_cells,
        estimated_cost_usd=readiness.estimated_cost_usd,
    )


def analyze_comparison_rows(
    *,
    comparison_id: str,
    preview_digest: str,
    rows: Sequence[Mapping[str, Any]],
    source: str,
) -> ComparisonResultV1:
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ValueError("comparison result requires at least one attempt row")
    pairs: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    baseline_passed = 0
    candidate_passed = 0
    for row in normalized:
        variant = str(row.get("variant_id") or "")
        if variant not in {"baseline", "candidate"}:
            continue
        task = str(row.get("task_id") or row.get("task_name") or "")
        harness = str(row.get("harness") or "")
        attempt = int(row.get("trial_index") or 1)
        pairs.setdefault((task, harness, attempt), {})[variant] = row
        baseline_passed += variant == "baseline" and row.get("pass") is True
        candidate_passed += variant == "candidate" and row.get("pass") is True
    paired_cases: list[dict[str, Any]] = []
    improved = regressed = unchanged = incomplete = 0
    for (task, harness, attempt), pair in sorted(pairs.items()):
        base = pair.get("baseline")
        candidate = pair.get("candidate")
        if (
            base is None
            or candidate is None
            or base.get("wandb_serverless_eligible") is False
            or candidate.get("wandb_serverless_eligible") is False
        ):
            status = "incomplete"
            incomplete += 1
        elif base.get("pass") is False and candidate.get("pass") is True:
            status = "improved"
            improved += 1
        elif base.get("pass") is True and candidate.get("pass") is False:
            status = "regressed"
            regressed += 1
        else:
            status = "unchanged"
            unchanged += 1
        paired_cases.append(
            {
                "task_id": task,
                "harness": harness,
                "attempt": attempt,
                "status": status,
                "baseline_prediction_id": (
                    str(base.get("prediction_id") or "") if base else None
                ),
                "candidate_prediction_id": (
                    str(candidate.get("prediction_id") or "") if candidate else None
                ),
            }
        )
    limitations = [
        "Task outcomes and authored evaluations must be interpreted separately.",
        "The result applies only to the locked taskset, candidates, and attempts.",
    ]
    if incomplete:
        limitations.append("At least one aligned pair is incomplete.")
    unsigned = ComparisonResultV1(
        schema_version=COMPARISON_SCHEMA_VERSION,
        comparison_id=validate_id(comparison_id, kind="comparison id"),
        preview_digest=preview_digest,
        source=source,
        rows=len(normalized),
        baseline_passed=baseline_passed,
        candidate_passed=candidate_passed,
        improved=improved,
        regressed=regressed,
        unchanged=unchanged,
        incomplete=incomplete,
        required_evaluations_incomplete=sum(
            1
            for row in normalized
            if row.get("comparison_required_evaluation_complete") is False
        ),
        deterministic_summary=_deterministic_summary(normalized),
        judge_summary=_judge_summary(normalized),
        mechanism_summary=_mechanism_summary(normalized),
        operational_summary=_operational_summary(normalized),
        evidence_links=_comparison_evidence_links(normalized),
        paired_cases=tuple(paired_cases),
        limitations=tuple(limitations),
    )
    return replace(
        unsigned,
        result_digest=_artifact_digest(unsigned.to_dict(), "result_digest"),
    )


def write_comparison_result(
    result: ComparisonResultV1, *, destination: Path
) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "result.json"
    markdown_path = destination / "result.md"
    atomic_write_json(json_path, result.to_dict())
    _atomic_text(markdown_path, _result_markdown(result))
    artifacts = {
        path.name: _sha256_path(path)
        for path in (
            json_path,
            markdown_path,
            destination / "attempts.jsonl",
        )
        if path.is_file()
    }
    atomic_write_json(
        destination / "reproduction.json",
        {
            "schema_version": 1,
            "comparison_id": result.comparison_id,
            "preview_digest": result.preview_digest,
            "result_digest": result.result_digest,
            "source": result.source,
            "artifacts": artifacts,
            "private_labels_included": False,
            "commands": {
                "inspect": f"uv run fugue result {result.comparison_id}",
                "replay": (
                    "uv run fugue demo source-use"
                    if result.source == "bundled-replay"
                    else None
                ),
            },
            "limitations": [
                "Private labels are intentionally excluded.",
                "A live rerun requires the original locked private labels, "
                "component locks, runtime locks, and exact-preview approval.",
            ],
        },
    )
    return json_path, markdown_path


def scaffold_comparison(destination: Path, *, force: bool = False) -> Path:
    root = destination.resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(
            f"refusing to overwrite non-empty comparison directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    skill_root = (
        root / "configs" / "fugue" / "skills" / "verify-current-source"
    )
    skill_root.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        root / "tasks.jsonl",
        json.dumps(
            {
                "id": "policy-limit",
                "input": {
                    "question": (
                        "Find the current expense limit and return JSON with "
                        "amount and source."
                    )
                },
                "tags": ["source-use"],
                "partition": "holdout",
            },
            sort_keys=True,
        )
        + "\n",
    )
    _atomic_text(
        root / "private-labels.jsonl",
        json.dumps(
            {
                "id": "policy-limit",
                "expected": {"amount": 125, "source": "expense-policy-v4.md"},
                "base_output": {
                    "amount": 100,
                    "source": "expense-policy-v3.md",
                },
                "gold_output": {
                    "amount": 125,
                    "source": "expense-policy-v4.md",
                },
            },
            sort_keys=True,
        )
        + "\n",
    )
    _atomic_text(
        skill_root / "SKILL.md",
        (
            "---\n"
            "name: verify-current-source\n"
            "description: Inspect and cite the current authoritative source.\n"
            "---\n\n"
            "# Verify current source\n\n"
            "Open the authoritative source before answering. Prefer a current, "
            "effective document over a draft or superseded revision, and cite "
            "the exact filename used.\n"
        ),
    )
    config = {
        "schema_version": 1,
        "id": "source-use",
        "question": "Does verifying the current source improve evidence use?",
        "taskset": {
            "tasks": "tasks.jsonl",
            "private_labels": "private-labels.jsonl",
        },
        "baseline": {"label": "Current Agent"},
        "candidate": {
            "label": "Current Agent + source verification Skill",
            "skills": ["verify-current-source"],
        },
        "changed": ["skills"],
        "evaluators": [
            {
                "id": "fact-and-source",
                "type": "deterministic",
                "required": True,
                "checks": ["answer_present", "expected_values"],
            }
        ],
        "execution": {
            "model": "wandb/zai-org/GLM-5.2",
            "harnesses": ["codex"],
            "attempts": 2,
            "concurrency": 1,
            "max_cost_usd": 40,
            "reserve_per_attempt_usd": 10,
            "approval_required": True,
            "trace_content": "full",
        },
    }
    _atomic_text(
        root / "comparison.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )
    _atomic_text(
        root / "README.md",
        (
            "# Fugue Agent-change comparison\n\n"
            "Run from the repository root:\n\n"
            "```bash\n"
            f"uv run fugue check {root.as_posix()}/comparison.yaml\n"
            "```\n"
        ),
    )
    return root / "comparison.yaml"


def score_comparison_rows(
    spec: ComparisonSpecV1,
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    env: Mapping[str, str] | None = None,
    judge_request: Callable[..., tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
    | None = None,
) -> list[dict[str, Any]]:
    public_tasks = {
        str(item["id"]): item
        for item in _load_public_tasks(repo_root / spec.taskset.tasks)
    }
    labels = {
        str(item["id"]): item
        for item in _load_private_labels(repo_root / spec.taskset.private_labels)
    }
    deterministic = tuple(
        evaluator
        for evaluator in spec.evaluators
        if evaluator.type == "deterministic"
    )
    scored: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        task_id = str(row.get("task_id") or row.get("task_name") or "")
        output = (
            row.get("final_output")
            if row.get("final_output") is not None
            else row.get("answer")
        )
        label = labels.get(task_id)
        row.setdefault("benchmark_pass", row.get("pass"))
        if label is None:
            row["pass"] = None
            row["comparison_evaluation_status"] = "unavailable"
            row["comparison_evaluation_reason"] = "private task label is unavailable"
            row["comparison_required_evaluation_complete"] = False
        else:
            try:
                passed, dimensions = _score_deterministic_output(
                    task=public_tasks.get(task_id, {}),
                    output=output,
                    expected=label["expected"],
                    evidence=_custom_scorer_evidence(row),
                    evaluators=deterministic,
                    repo_root=repo_root,
                )
            except Exception as exc:
                row["pass"] = None
                row["comparison_evaluation_status"] = "unavailable"
                row["comparison_evaluation_reason"] = (
                    "deterministic evaluation failed: "
                    f"{type(exc).__name__}"
                )
                row["comparison_required_evaluation_complete"] = False
            else:
                row["pass"] = passed
                row["comparison_evaluation_status"] = "scored"
                row["comparison_deterministic_scores"] = dimensions
                row["comparison_mechanism"] = _comparison_mechanism(
                    row,
                    expected=label["expected"],
                    passed=passed,
                    candidate_skill_ids=spec.candidate.skills,
                )
                row["comparison_required_evaluation_complete"] = True
        judge_results: dict[str, Any] = {}
        judge_scores: dict[str, float] = {}
        for judge in (
            evaluator
            for evaluator in spec.evaluators
            if evaluator.type == "llm_judge"
        ):
            if env is None:
                judge_results[judge.id] = {
                    "status": "unavailable",
                    "reason": "judge execution was not requested",
                }
                if judge.required:
                    row["comparison_required_evaluation_complete"] = False
                continue
            try:
                request = judge_request or _request_comparison_judge
                payload, usage, receipt = request(
                    evaluator=judge,
                    public_task=public_tasks.get(task_id, {}),
                    row=row,
                    env=env,
                )
                parsed = _validate_comparison_judge_payload(judge, payload)
                for dimension, value in parsed["scores"].items():
                    judge_scores[f"{judge.id}.{dimension}"] = value
                judge_results[judge.id] = {
                    "status": "scored",
                    **parsed,
                    "usage": usage,
                    "route_receipt": receipt,
                    "cost_usd": None,
                }
            except Exception as exc:
                judge_results[judge.id] = {
                    "status": "unavailable",
                    "reason": (
                        "judge evaluation failed: "
                        f"{type(exc).__name__}"
                    ),
                }
                if judge.required:
                    row["comparison_required_evaluation_complete"] = False
        if judge_results:
            row["comparison_judges"] = judge_results
            row["comparison_judge_status"] = (
                "scored"
                if all(value["status"] == "scored" for value in judge_results.values())
                else "unavailable"
            )
        if judge_scores:
            row["comparison_judge_scores"] = judge_scores
        scored.append(row)
    return scored


def _request_comparison_judge(
    *,
    evaluator: ComparisonEvaluatorV1,
    public_task: Mapping[str, Any],
    row: Mapping[str, Any],
    env: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from fugue.bench.evaluations import _post_judge
    from fugue.model_plane import (
        model_route_identity,
        provider_api_key,
        provider_api_key_env,
        resolve_model_route,
    )

    if not evaluator.profile or not evaluator.rubric:
        raise ValueError("comparison judge is missing its profile or public rubric")
    route = resolve_model_route(evaluator.profile, env)
    api_key = provider_api_key(route, env)
    if not api_key:
        raise RuntimeError(
            f"{provider_api_key_env(route)} is required for comparison judging"
        )
    permitted = _comparison_judge_evidence(row, evaluator.evidence)
    payload = {
        "public_task": {
            "input": public_task.get("input"),
            "tags": public_task.get("tags") or [],
        },
        "final_response": _comparison_output(row),
        "permitted_evidence": permitted,
        "rubric": evaluator.rubric,
        "dimensions": list(evaluator.dimensions),
    }
    prompt = (
        "Blindly evaluate one Agent attempt. You do not know whether it came from "
        "the baseline or candidate. Use only the supplied public task, final "
        "response, permitted evidence, and rubric. Return one JSON object with: "
        "scores (one 0..1 number per dimension), overall_assessment (brief text), "
        "uncertainty (0..1), and rationale (at most 500 characters). Do not return "
        "hidden reasoning or a chain of thought.\n\n"
        + json.dumps(payload, sort_keys=True, default=str)[:48_000]
    )
    import httpx

    with httpx.Client(timeout=120) as client:
        response, usage = _post_judge(client, route, api_key, env, prompt)
    return (
        response,
        usage,
        {
            "schema_version": 1,
            "role": "blind_comparison_judge",
            "judge_id": evaluator.id,
            "profile": evaluator.profile,
            "route": model_route_identity(route, env),
            "rubric_digest": _judge_contract_digest(evaluator),
            "blind_fields": [
                "baseline_or_candidate",
                "candidate_revision",
                "variant_id",
                "treatment",
                "harness",
                "model",
                "deterministic_scores",
                "private_expected_values",
                "receipts",
                "internal_ids",
            ],
            "usage": usage,
        },
    )


def _comparison_judge_evidence(
    row: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_name in fields:
        if field_name == "tool_names":
            names = sorted(
                {
                    str(item.get("name") or item.get("tool") or "")
                    for item in row.get("tool_calls") or []
                    if isinstance(item, Mapping)
                    and (item.get("name") or item.get("tool"))
                }
            )
            result[field_name] = names
        elif field_name in {
            "artifact_paths",
            "retrieved_paths",
            "inspected_paths",
            "changed_paths",
        }:
            values = row.get(field_name)
            if isinstance(values, list):
                result[field_name] = [str(value)[:500] for value in values[:100]]
    return result


def _comparison_output(row: Mapping[str, Any]) -> Any:
    return (
        row.get("final_output")
        if row.get("final_output") is not None
        else row.get("answer")
    )


def _validate_comparison_judge_payload(
    evaluator: ComparisonEvaluatorV1, payload: Mapping[str, Any]
) -> dict[str, Any]:
    expected = set(evaluator.dimensions)
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, Mapping) or set(raw_scores) != expected:
        raise ValueError("judge scores do not match the locked rubric dimensions")
    scores: dict[str, float] = {}
    for dimension, raw in raw_scores.items():
        if (
            not isinstance(raw, int | float)
            or isinstance(raw, bool)
            or not 0 <= float(raw) <= 1
        ):
            raise ValueError(f"judge score {dimension!r} must be between zero and one")
        scores[str(dimension)] = float(raw)
    assessment = str(payload.get("overall_assessment") or "").strip()
    rationale = str(payload.get("rationale") or "").strip()
    uncertainty = payload.get("uncertainty")
    if not assessment or len(assessment) > 500:
        raise ValueError("judge overall_assessment must be 1..500 characters")
    if not rationale or len(rationale) > 500:
        raise ValueError("judge rationale must be 1..500 characters")
    if (
        not isinstance(uncertainty, int | float)
        or isinstance(uncertainty, bool)
        or not 0 <= float(uncertainty) <= 1
    ):
        raise ValueError("judge uncertainty must be between zero and one")
    return {
        "scores": scores,
        "overall_assessment": assessment,
        "uncertainty": float(uncertainty),
        "rationale": rationale,
    }


def _comparison_mechanism(
    row: Mapping[str, Any],
    *,
    expected: Any,
    passed: bool,
    candidate_skill_ids: tuple[str, ...],
) -> dict[str, str]:
    variant = str(row.get("variant_id") or "")
    skill_applicable = variant == "candidate" and bool(candidate_skill_ids)
    assigned = {
        str(value)
        for value in (
            row.get("skills_assigned") or row.get("skill_ids") or []
        )
    }
    registered = {str(value) for value in row.get("skills_registered") or []}
    registration_status = str(row.get("skill_registration_status") or "")
    invocation = row.get("skill_invocation_evidence") or {}
    invocation_status = (
        str(invocation.get("status") or "")
        if isinstance(invocation, Mapping)
        else ""
    )
    invoked = (
        {str(value) for value in invocation.get("skills_invoked") or []}
        if isinstance(invocation, Mapping)
        else set()
    )
    expected_skills = set(candidate_skill_ids)
    source = (
        str(expected.get("source") or expected.get("source_document") or "")
        if isinstance(expected, Mapping)
        else ""
    )
    returned_paths = _row_paths(
        row,
        "context_result_paths",
        "retrieved_paths",
        "search_result_paths",
    )
    opened_paths = _row_paths(
        row,
        "inspected_paths",
        "opened_paths",
        "context_result_opened_paths",
    )
    source_returned = _path_observed(source, returned_paths)
    source_opened = _path_observed(source, opened_paths)
    output = (
        row.get("final_output")
        if row.get("final_output") is not None
        else row.get("answer")
    )
    parsed = json.loads(output) if isinstance(output, str) and _is_json(output) else output
    source_used = bool(
        source
        and source_opened
        and isinstance(parsed, Mapping)
        and (
            parsed.get("source") == source
            or parsed.get("source_document") == source
        )
    )
    return {
        "skill_assigned": _mechanism_state(
            applicable=skill_applicable,
            available=bool(assigned) or not skill_applicable,
            reached=expected_skills <= assigned,
        ),
        "skill_registered": _mechanism_state(
            applicable=skill_applicable,
            available=registration_status
            not in {"", "unavailable"}
            or bool(registered),
            reached=(
                registration_status == "registered"
                and (not registered or expected_skills <= registered)
            ),
        ),
        "skill_invoked": _mechanism_state(
            applicable=skill_applicable,
            available=invocation_status
            not in {"", "unavailable"},
            reached=invocation_status == "observed"
            and expected_skills <= invoked,
        ),
        "relevant_source_returned": _mechanism_state(
            applicable=bool(returned_paths),
            available=bool(returned_paths),
            reached=source_returned,
        ),
        "relevant_source_opened": _mechanism_state(
            applicable=bool(source),
            available=bool(opened_paths),
            reached=source_opened,
        ),
        "relevant_source_used": _mechanism_state(
            applicable=bool(source),
            available=bool(opened_paths) and output is not None,
            reached=source_used,
        ),
        "task_passed": "observed" if passed else "not_observed",
    }


def _mechanism_state(
    *, applicable: bool, available: bool, reached: bool
) -> str:
    if not applicable:
        return "not_applicable"
    if not available:
        return "unavailable"
    return "observed" if reached else "not_observed"


def _row_paths(row: Mapping[str, Any], *keys: str) -> set[str]:
    result: set[str] = set()
    for key in keys:
        value = row.get(key) or []
        if isinstance(value, str):
            result.add(value)
        elif isinstance(value, list | tuple | set):
            result.update(str(item) for item in value)
    return result


def _path_observed(expected: str, observed: set[str]) -> bool:
    return bool(
        expected
        and any(
            value == expected or PurePosixPath(value).name == PurePosixPath(expected).name
            for value in observed
        )
    )


def _deterministic_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in ("baseline", "candidate"):
        selected = [
            row for row in rows if str(row.get("variant_id") or "") == variant
        ]
        dimensions = sorted(
            {
                str(key)
                for row in selected
                for key in (
                    row.get("comparison_deterministic_scores") or {}
                )
            }
        )
        result[variant] = {
            "passed": sum(row.get("pass") is True for row in selected),
            "evaluated": sum(
                row.get("comparison_evaluation_status") == "scored"
                for row in selected
            ),
            "dimensions": {
                dimension: {
                    "passed": sum(
                        _dimension_passed(
                            (row.get("comparison_deterministic_scores") or {}).get(
                                dimension
                            )
                        )
                        for row in selected
                    ),
                    "evaluated": sum(
                        dimension
                        in (row.get("comparison_deterministic_scores") or {})
                        for row in selected
                    ),
                    "mean": _numeric_summary(
                        [
                            (
                                float(value)
                                if isinstance(value, bool)
                                else value
                            )
                            for row in selected
                            if (
                                value := (
                                    row.get("comparison_deterministic_scores") or {}
                                ).get(dimension)
                            )
                            is not None
                        ]
                    )["mean"],
                }
                for dimension in dimensions
            },
        }
    return result


def _dimension_passed(value: Any) -> bool:
    return value is True or (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and float(value) == 1.0
    )


def _judge_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [
        row
        for row in rows
        if isinstance(row.get("comparison_judge_scores"), Mapping)
    ]
    if not scored:
        if any(row.get("comparison_judge_status") for row in rows):
            return {
                "status": "unavailable",
                "attempts": sum(
                    row.get("comparison_judge_status") == "unavailable"
                    for row in rows
                ),
            }
        return {"status": "not_used"}
    dimensions = sorted(
        {
            str(key)
            for row in scored
            for key in (row.get("comparison_judge_scores") or {})
        }
    )
    by_variant: dict[str, Any] = {}
    for variant in ("baseline", "candidate"):
        selected = [
            row for row in scored if str(row.get("variant_id") or "") == variant
        ]
        by_variant[variant] = {
            dimension: _numeric_summary(
                [
                    (row.get("comparison_judge_scores") or {}).get(dimension)
                    for row in selected
                ]
            )
            for dimension in dimensions
        }
    return {
        "status": "scored",
        "by_variant": by_variant,
        "unavailable_attempts": sum(
            row.get("comparison_judge_status") == "unavailable"
            for row in rows
        ),
    }


def _numeric_summary(values: Sequence[Any]) -> dict[str, Any]:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return {
        "evaluated": len(numeric),
        "mean": round(sum(numeric) / len(numeric), 6) if numeric else None,
    }


def _mechanism_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stages = sorted(
        {
            str(key)
            for row in rows
            for key in (row.get("comparison_mechanism") or {})
        }
    )
    return {
        stage: {
            variant: {
                "observed": sum(
                    (row.get("comparison_mechanism") or {}).get(stage)
                    == "observed"
                    for row in rows
                    if str(row.get("variant_id") or "") == variant
                ),
                "applicable": sum(
                    (row.get("comparison_mechanism") or {}).get(stage)
                    not in {None, "not_applicable"}
                    for row in rows
                    if str(row.get("variant_id") or "") == variant
                ),
                "unavailable": sum(
                    (row.get("comparison_mechanism") or {}).get(stage)
                    == "unavailable"
                    for row in rows
                    if str(row.get("variant_id") or "") == variant
                ),
            }
            for variant in ("baseline", "candidate")
        }
        for stage in stages
    }


def _comparison_evidence_links(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    fields = {
        "agent_url": "Agent trace",
        "weave_agent_url": "Agent trace",
        "evaluation_url": "Evaluation",
        "weave_evaluation_url": "Evaluation",
        "dataset_url": "Dataset",
        "weave_dataset_url": "Dataset",
    }
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        for field_name, label in fields.items():
            value = str(row.get(field_name) or "")
            if not _safe_evidence_url(value) or value in seen:
                continue
            seen.add(value)
            result.append({"label": label, "url": value})
    return tuple(result)


def _safe_evidence_url(value: str) -> bool:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme == "https" and parsed.netloc:
        return True
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}


def _operational_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    execution: dict[str, int] = {}
    evidence: dict[str, int] = {}
    observed_cost = 0.0
    cost_rows = 0
    latency_ms = 0.0
    latency_rows = 0
    input_tokens = 0
    output_tokens = 0
    usage_rows = 0
    infrastructure_failures = 0
    wandb_rows = 0
    wandb_eligible = 0
    for row in rows:
        status = str(
            row.get("status")
            or row.get("execution_status")
            or "unknown"
        )
        execution[status] = execution.get(status, 0) + 1
        evidence_status = str(
            row.get("trace_link_status")
            or row.get("evidence_status")
            or "unknown"
        )
        evidence[evidence_status] = evidence.get(evidence_status, 0) + 1
        if status in {"failed", "error", "infrastructure_failed"} or row.get(
            "exception_class"
        ):
            infrastructure_failures += 1
        if "wandb_serverless_eligible" in row:
            wandb_rows += 1
            wandb_eligible += row.get("wandb_serverless_eligible") is True
        cost = row.get("accounted_cost_usd", row.get("cost_usd"))
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            observed_cost += float(cost)
            cost_rows += 1
        latency = row.get("latency_ms")
        if isinstance(latency, int | float) and not isinstance(latency, bool):
            latency_ms += float(latency)
            latency_rows += 1
        row_input = row.get("input_tokens")
        row_output = row.get("output_tokens")
        if isinstance(row_input, int) and isinstance(row_output, int):
            input_tokens += row_input
            output_tokens += row_output
            usage_rows += 1
    result = {
        "execution_states": dict(sorted(execution.items())),
        "evidence_states": dict(sorted(evidence.items())),
        "infrastructure_failures": infrastructure_failures,
        "observed_cost_usd": round(observed_cost, 6) if cost_rows else None,
        "cost_rows": cost_rows,
        "latency_ms": round(latency_ms, 3) if latency_rows else None,
        "latency_rows": latency_rows,
        "input_tokens": input_tokens if usage_rows else None,
        "output_tokens": output_tokens if usage_rows else None,
        "usage_rows": usage_rows,
    }
    if wandb_rows:
        result["wandb_serverless"] = {
            "rows": wandb_rows,
            "eligible": wandb_eligible,
            "ineligible": wandb_rows - wandb_eligible,
        }
    return result


def _comparison_scorer_names(spec: ComparisonSpecV1) -> tuple[str, ...]:
    names: set[str] = set()
    for evaluator in spec.evaluators:
        if evaluator.type == "llm_judge":
            names.update(
                f"comparison.judge.{evaluator.id}.{dimension}"
                for dimension in evaluator.dimensions
            )
        elif evaluator.scorer:
            names.update(
                f"comparison.deterministic.{evaluator.id}.{dimension}"
                for dimension in evaluator.dimensions
            )
        else:
            names.update(
                f"comparison.deterministic.{check}"
                for check in evaluator.checks
            )
    return tuple(sorted(names))


def execute_comparison(
    preview: ComparisonPreviewV1,
    *,
    approval_digest: str,
    repo_root: Path,
    env_file: Path | None = None,
    fetch_weave: bool = True,
) -> tuple[ComparisonResultV1, Path, Path]:
    _verify_artifact(preview.to_dict(), "preview_digest", "comparison preview")
    spec = comparison_from_dict(
        preview.comparison,
        repo_root=repo_root,
        source=repo_root,
    )
    if spec.execution.approval_required:
        if not approval_digest:
            raise ValueError("comparison execution requires an approval digest")
        claim_comparison_approval(
            preview,
            approval_digest=approval_digest,
            repo_root=repo_root,
        )
    experiment, request = materialize_comparison(preview, repo_root=repo_root)
    service = OperatorService(repo_root, env_file)
    service.prepare(request, experiment=experiment)
    from fugue.bench.execution import new_run_id

    run_id = new_run_id()

    def evaluate_attempt(row: dict[str, Any]) -> None:
        scored = score_comparison_rows(
            spec,
            [row],
            repo_root=repo_root,
            env=service.env,
        )[0]
        row.update(scored)

    service.execute_run(
        request,
        run_id=run_id,
        experiment=experiment,
        host_evaluator=evaluate_attempt,
        host_scorer_names=_comparison_scorer_names(spec),
    )
    export_path = (
        repo_root / COMPARISON_RESULT_ROOT / preview.preview_digest / "attempts.jsonl"
    )
    summary = service.export_run(
        run_id,
        out=export_path,
        fetch_weave=fetch_weave,
        to_weave=False,
    )
    rows = _read_jsonl(summary.path, "comparison attempt rows")
    scored = []
    for row in rows:
        if row.get("comparison_evaluation_status") in {"scored", "unavailable"}:
            scored.append(row)
        else:
            scored.extend(
                score_comparison_rows(
                    spec,
                    [row],
                    repo_root=repo_root,
                    env=service.env,
                )
            )
    _atomic_text(summary.path, _jsonl(scored))
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=scored,
        source=run_id,
    )
    destination = repo_root / COMPARISON_RESULT_ROOT / preview.preview_digest
    json_path, markdown_path = write_comparison_result(
        result, destination=destination
    )
    atomic_write_json(
        repo_root / COMPARISON_RESULT_ROOT / "latest.json",
        {
            "comparison_id": spec.id,
            "preview_digest": preview.preview_digest,
            "result": json_path.relative_to(repo_root).as_posix(),
            "markdown": markdown_path.relative_to(repo_root).as_posix(),
        },
    )
    return result, json_path, markdown_path


def _candidate(raw: Any, default_label: str) -> ComparisonCandidateV1:
    value = _mapping(raw, "candidate")
    _reject_unknown(
        value,
        {
            "label",
            "prompt_id",
            "skills",
            "context",
            "integrations",
            "agent_kwargs",
            "environment",
        },
        "candidate",
    )
    skills = _string_tuple(value.get("skills") or [], "skill", allow_empty=True)
    for item in skills:
        validate_id(item, kind="skill id")
    if len(set(skills)) != len(skills):
        raise ValueError("candidate skills must be unique")
    context = dict(
        value.get("context")
        or {"system_id": "none", "delivery": "portable"}
    )
    _reject_unknown(context, {"system_id", "delivery", "config"}, "candidate context")
    if context.get("delivery") not in {"portable", "native_mcp"}:
        raise ValueError("candidate context delivery must be portable or native_mcp")
    integrations = tuple(
        _integration(value, index)
        for index, value in enumerate(
            _sequence(value.get("integrations") or [], "integrations", allow_empty=True),
            start=1,
        )
    )
    if len({item["id"] for item in integrations}) != len(integrations):
        raise ValueError("candidate integrations must be unique")
    return ComparisonCandidateV1(
        label=_text(value.get("label") or default_label, "candidate label", 200),
        prompt_id=(
            validate_id(str(value["prompt_id"]), kind="prompt id")
            if value.get("prompt_id")
            else None
        ),
        skills=skills,
        context=context,
        integrations=integrations,
        agent_kwargs=dict(_mapping(value.get("agent_kwargs") or {}, "agent kwargs")),
        environment=dict(_mapping(value.get("environment") or {}, "environment")),
    )


def _integration(raw: Any, index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"id": validate_id(raw, kind="integration id")}
    value = _mapping(raw, f"integration {index}")
    _reject_unknown(value, {"id", "config"}, f"integration {index}")
    return _drop_empty(
        {
            "id": validate_id(str(value.get("id") or ""), kind="integration id"),
            "config": dict(_mapping(value.get("config") or {}, "integration config")),
        }
    )


def _evaluator(raw: Any) -> ComparisonEvaluatorV1:
    value = _mapping(raw, "evaluator")
    _reject_unknown(
        value,
        {
            "id",
            "type",
            "required",
            "checks",
            "scorer",
            "runtime",
            "profile",
            "calibration",
            "rubric",
            "dimensions",
            "evidence",
            "reserve_cost_usd",
        },
        "evaluator",
    )
    evaluator_type = str(value.get("type") or "")
    if evaluator_type not in {"deterministic", "llm_judge"}:
        raise ValueError("evaluator type must be deterministic or llm_judge")
    checks = _string_tuple(
        value.get("checks") or [],
        "evaluator check",
        allow_empty=True,
    )
    profile = str(value.get("profile") or "") or None
    scorer = str(value.get("scorer") or "") or None
    runtime = str(value.get("runtime") or "") or None
    calibration = str(value.get("calibration") or "") or None
    rubric = str(value.get("rubric") or "").strip() or None
    dimensions = _string_tuple(
        value.get("dimensions") or [], "judge dimension", allow_empty=True
    )
    evidence = _string_tuple(
        value.get("evidence") or [], "judge evidence", allow_empty=True
    )
    if evaluator_type == "deterministic" and bool(checks) == bool(scorer):
        raise ValueError(
            "deterministic evaluator requires exactly one of checks or scorer"
        )
    if evaluator_type == "deterministic" and scorer and not runtime:
        runtime = "python312-sandbox-v1"
    if evaluator_type == "deterministic" and runtime and not scorer:
        raise ValueError("deterministic evaluator runtime requires scorer")
    if evaluator_type == "deterministic" and scorer:
        validate_id(str(runtime), kind="scorer runtime id")
        if not dimensions:
            raise ValueError("custom deterministic scorer requires dimensions")
    if evaluator_type == "llm_judge" and not profile:
        raise ValueError("LLM judge evaluator requires a profile")
    if evaluator_type == "llm_judge" and not rubric:
        raise ValueError("LLM judge evaluator requires a public rubric")
    if evaluator_type == "llm_judge" and not dimensions:
        raise ValueError("LLM judge evaluator requires dimensions")
    unsupported_evidence = sorted(
        set(evidence)
        - {
            "tool_names",
            "artifact_paths",
            "retrieved_paths",
            "inspected_paths",
            "changed_paths",
        }
    )
    if unsupported_evidence:
        raise ValueError(
            "unsupported blind-judge evidence fields: "
            + ", ".join(unsupported_evidence)
        )
    return ComparisonEvaluatorV1(
        id=validate_id(str(value.get("id") or ""), kind="evaluator id"),
        type=evaluator_type,  # type: ignore[arg-type]
        required=bool(value.get("required", True)),
        checks=checks,
        scorer=scorer,
        runtime=runtime,
        profile=profile,
        calibration=calibration,
        rubric=rubric,
        dimensions=dimensions,
        evidence=evidence,
        reserve_cost_usd=_non_negative_number(
            value.get("reserve_cost_usd", 0), "judge reserve"
        ),
    )


def _execution(raw: Any) -> ComparisonExecutionPolicyV1:
    value = _mapping(raw, "execution policy")
    _reject_unknown(
        value,
        {
            "model",
            "harnesses",
            "attempts",
            "concurrency",
            "max_cost_usd",
            "reserve_per_attempt_usd",
            "approval_required",
            "trace_content",
            "environment",
        },
        "execution policy",
    )
    harnesses = _string_tuple(value.get("harnesses"), "harness")
    unknown = sorted(set(harnesses) - set(_HARNESS_AGENTS))
    if unknown:
        raise ValueError("unsupported harnesses: " + ", ".join(unknown))
    trace_content = str(value.get("trace_content") or "full")
    if trace_content not in {"full", "metadata"}:
        raise ValueError("trace_content must be full or metadata")
    return ComparisonExecutionPolicyV1(
        model=_text(value.get("model"), "execution model", 300),
        harnesses=harnesses,
        attempts=_positive_int(value.get("attempts", 1), "attempts"),
        concurrency=_positive_int(value.get("concurrency", 1), "concurrency"),
        max_cost_usd=_non_negative_number(
            value.get("max_cost_usd", 0), "maximum cost"
        ),
        reserve_per_attempt_usd=_non_negative_number(
            value.get("reserve_per_attempt_usd", 0),
            "attempt reserve",
        ),
        approval_required=bool(value.get("approval_required", True)),
        trace_content=trace_content,  # type: ignore[arg-type]
        environment=dict(
            _mapping(value.get("environment") or {}, "execution environment")
        ),
    )


def _public_case(
    task: Mapping[str, Any],
    *,
    spec: ComparisonSpecV1,
    index: int,
    repo_root: Path,
) -> dict[str, Any]:
    task_id = str(task["id"])
    input_value = task["input"]
    instruction = (
        str(input_value["question"])
        if isinstance(input_value, dict) and isinstance(input_value.get("question"), str)
        else json.dumps(input_value, indent=2, sort_keys=True)
    )
    applicability = {
        harness: {"applicable": True, "reason": None}
        for harness in spec.execution.harnesses
    }
    return {
        "schema_version": 1,
        "id": task_id,
        "title": task_id.replace("-", " ").title(),
        "instruction": instruction,
        "attachments": _task_attachments(task, repo_root),
        "environment": {
            "profile_id": "artifact-python-v1",
            "profile_digest": stable_digest(
                {
                    "id": "artifact-python-v1",
                    "image": _COMPARISON_BASE_IMAGE,
                }
            ),
            "kind": "artifact",
            "base_image": _COMPARISON_BASE_IMAGE,
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 10240,
            "repository": None,
            "integration_ids": sorted(
                {
                    str(item["id"])
                    for candidate in (spec.baseline, spec.candidate)
                    for item in candidate.integrations
                }
            ),
        },
        "interaction": {
            "type": "single_turn",
            "profile_id": None,
            "scripted_turns": [],
            "directions": [],
            "max_user_turns": 0,
            "max_agent_turns": 1,
            "timeout_sec": 900,
            "controller_digest": stable_digest(
                {"type": "single_turn", "max_agent_turns": 1}
            ),
        },
        "harness_applicability": applicability,
        "profile_digests": {},
        "scenario_id": "comparison",
        "tags": list(task.get("tags") or []),
        "partition": str(task.get("partition") or "holdout"),
        "source_index": index,
        "task_definition_digest": stable_digest(
            {
                "comparison_id": spec.id,
                "task_id": task_id,
                "public_task": dict(task),
            }
        ),
    }


def _task_attachments(
    task: Mapping[str, Any], repo_root: Path
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(task.get("resources") or [], start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"public task {task['id']} resource {index} must be an object"
            )
        _reject_unknown(
            raw,
            {"path", "target"},
            f"public task {task['id']} resource {index}",
        )
        relative = _safe_resource_relative_path(
            raw.get("path"),
            label=f"public task {task['id']} resource {index}",
        )
        source = repo_root / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(
                f"public task {task['id']} resource is not a regular file: {relative}"
            )
        target = str(raw.get("target") or "")
        target_path = PurePosixPath(target)
        allowed_root = PurePosixPath("/workspace/resources")
        if (
            not target_path.is_absolute()
            or any(part in {"", ".", ".."} for part in target_path.parts)
            or target_path.parts[: len(allowed_root.parts)] != allowed_root.parts
        ):
            raise ValueError(
                f"public task {task['id']} resource target must be under "
                "/workspace/resources"
            )
        result.append(
            {
                "locked_relative": relative,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "target": target_path.as_posix(),
            }
        )
    return result


def _safe_resource_relative_path(value: Any, *, label: str) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} path must be a safe repository-relative file")
    return path.as_posix()


def _variant_dict(variant_id: str, value: ComparisonCandidateV1) -> dict[str, Any]:
    return {
        "id": variant_id,
        "label": value.label,
        "prompt_id": value.prompt_id,
        "skills": list(value.skills),
        "context": value.context,
        "integrations": list(value.integrations),
        "agent_kwargs": value.agent_kwargs,
        "environment": value.environment,
    }


def _research_scorer(
    value: ComparisonEvaluatorV1,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    if value.type == "deterministic":
        if value.scorer:
            return {
                "id": value.id,
                "label": value.id.replace("-", " ").title(),
                "kind": "deterministic",
                "description": "Pinned custom scorer executed in an isolated runtime.",
                "required": value.required,
                "aggregation": "Every returned dimension must pass.",
                "revision": revision,
                "evidence_inputs": [
                    "Public task",
                    "Agent output",
                    "Permitted normalized evidence",
                    "Host-only expected values",
                ],
                "dimensions": [
                    {
                        "id": dimension.replace("_", "-"),
                        "label": dimension.replace("_", " ").title(),
                        "source_key": (
                            f"comparison.deterministic.{value.id}.{dimension}"
                        ),
                        "target": True,
                        "primary": index == 0,
                    }
                    for index, dimension in enumerate(value.dimensions)
                ],
            }
        return {
            "id": value.id,
            "label": value.id.replace("-", " ").title(),
            "kind": "deterministic",
            "description": "Deterministic checks compiled from the comparison evaluator.",
            "required": value.required,
            "aggregation": "Every required check must pass.",
            "revision": revision,
            "evidence_inputs": ["Agent output", "Host-only expected values"],
            "dimensions": [
                {
                    "id": check.replace("_", "-"),
                    "label": check.replace("_", " ").title(),
                    "source_key": check,
                    "target": True,
                    "primary": check == "expected_values",
                }
                for check in value.checks
            ],
        }
    return {
        "id": value.id,
        "label": value.id.replace("-", " ").title(),
        "kind": "llm_judge",
        "description": "Calibrated blind qualitative review.",
        "required": value.required,
        "revision": revision,
        "model": value.profile,
        "rubric_summary": "Use only permitted evidence and remain calibrated about coverage.",
        "dimensions": [
            {
                "id": dimension.replace("_", "-"),
                "label": dimension.replace("_", " ").title(),
                "source_key": dimension,
                "primary": dimension == "evidence_grounding",
            }
            for dimension in value.dimensions
        ],
        "blind_fields": [
            "harness",
            "model",
            "variant_id",
            "context_system_id",
            "candidate_id",
            "treatment",
        ],
        "evidence_inputs": ["Agent output", *value.evidence],
    }


def _load_public_tasks(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "public taskset")
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        _reject_unknown(row, _PUBLIC_TASK_FIELDS, f"public task {index}")
        task_id = validate_id(str(row.get("id") or ""), kind="task id")
        if task_id in ids:
            raise ValueError(f"duplicate public task id: {task_id}")
        ids.add(task_id)
        if "input" not in row:
            raise ValueError(f"public task {task_id} requires input")
        leaked = sorted(
            key for key in row if key.lower() in _PRIVATE_WORDS
        )
        if leaked:
            raise ValueError(
                f"public task {task_id} contains private field(s): "
                + ", ".join(leaked)
            )
        partition = str(row.get("partition") or "holdout")
        if partition not in {"qualification", "discovery", "holdout"}:
            raise ValueError(f"public task {task_id} has invalid partition")
        tags = row.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
            raise ValueError(f"public task {task_id} tags must be strings")
        resources = row.get("resources") or []
        if not isinstance(resources, list):
            raise ValueError(f"public task {task_id} resources must be an array")
        for resource_index, resource in enumerate(resources, start=1):
            if not isinstance(resource, dict):
                raise ValueError(
                    f"public task {task_id} resource {resource_index} must be an object"
                )
            _reject_unknown(
                resource,
                {"path", "target"},
                f"public task {task_id} resource {resource_index}",
            )
            _safe_resource_relative_path(
                resource.get("path"),
                label=f"public task {task_id} resource {resource_index}",
            )
        row["id"] = task_id
        row["partition"] = partition
        row["tags"] = tags
        row["resources"] = resources
    return rows


def _load_private_labels(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "private labels")
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        _reject_unknown(row, _PRIVATE_LABEL_FIELDS, f"private label {index}")
        task_id = validate_id(str(row.get("id") or ""), kind="task id")
        if task_id in ids:
            raise ValueError(f"duplicate private label id: {task_id}")
        if "expected" not in row:
            raise ValueError(f"private label {task_id} requires expected values")
        ids.add(task_id)
        row["id"] = task_id
    return rows


def _evaluator_digest(
    evaluator: ComparisonEvaluatorV1, repo_root: Path
) -> str:
    value = evaluator.to_dict()
    if evaluator.scorer:
        value["scorer_sha256"] = _sha256_path(
            _safe_input_path(
                Path(evaluator.scorer), repo_root, "deterministic scorer"
            )
        )
    if evaluator.calibration:
        value["calibration_sha256"] = _sha256_path(
            _safe_input_path(
                Path(evaluator.calibration), repo_root, "judge calibration"
            )
        )
    return stable_digest(value)


def _score_deterministic_output(
    *,
    task: Mapping[str, Any],
    output: Any,
    expected: Any,
    evidence: Mapping[str, Any],
    evaluators: Sequence[ComparisonEvaluatorV1],
    repo_root: Path,
) -> tuple[bool, dict[str, bool | float]]:
    scores: dict[str, bool | float] = {}
    evaluator_passes: list[bool] = []
    for evaluator in evaluators:
        if evaluator.scorer:
            payload = _run_custom_scorer(
                evaluator,
                task=task,
                output=output,
                evidence=evidence,
                expected=expected,
                repo_root=repo_root,
            )
            details = payload["details"]
            if not isinstance(details, Mapping) or not details:
                raise ValueError("custom scorer must return at least one dimension")
            if set(details) != set(evaluator.dimensions):
                raise ValueError(
                    "custom scorer output does not match its declared dimensions"
                )
            for name, value in details.items():
                dimension = str(name)
                if (
                    not 1 <= len(dimension) <= 100
                    or not dimension[0].isalnum()
                    or any(
                        not (character.isalnum() or character in {"_", "-"})
                        for character in dimension
                    )
                ):
                    raise ValueError(
                        "scorer dimension names must use letters, numbers, _ or -"
                    )
                if isinstance(value, bool):
                    normalized: bool | float = value
                elif (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and 0 <= float(value) <= 1
                ):
                    normalized = float(value)
                else:
                    raise ValueError(
                        f"scorer dimension {dimension!r} must be bool or 0..1"
                    )
                scores[f"{evaluator.id}.{dimension}"] = normalized
            evaluator_passes.append(float(payload["score"]) == 1.0)
            continue
        check_scores = {
            "answer_present": bool(
                output is not None
                and (not isinstance(output, str) or bool(output.strip()))
            ),
            "expected_values": _contains_expected(
                (
                    json.loads(output)
                    if isinstance(output, str) and _is_json(output)
                    else output
                ),
                expected,
            ),
        }
        selected = {
            check: check_scores[check]
            for check in evaluator.checks
        }
        scores.update(selected)
        evaluator_passes.append(all(selected.values()))
    return bool(evaluator_passes) and all(evaluator_passes), scores


def _run_custom_scorer(
    evaluator: ComparisonEvaluatorV1,
    *,
    task: Mapping[str, Any],
    output: Any,
    evidence: Mapping[str, Any],
    expected: Any,
    repo_root: Path,
) -> dict[str, Any]:
    from fugue.bench.task_authoring import (
        TaskAuthoringLimitsV1,
        load_task_profiles,
        run_inline_scorer,
    )

    if not evaluator.scorer or not evaluator.runtime:
        raise ValueError("custom scorer is missing its source or runtime")
    path = _safe_input_path(
        Path(evaluator.scorer), repo_root, "deterministic scorer"
    )
    source = path.read_text(encoding="utf-8")
    _validate_custom_scorer_source(source)
    wrapper = (
        source.rstrip()
        + "\n\n"
        + """
if __name__ == "__main__":
    import json
    import math
    import sys

    with open(sys.argv[1], encoding="utf-8") as handle:
        _fugue_payload = json.load(handle)
    _fugue_reference = _fugue_payload["reference"]
    _fugue_evidence = dict(_fugue_payload["evidence"])
    _fugue_evidence["expected"] = _fugue_reference["expected"]
    _fugue_result = score(
        _fugue_reference["task"],
        _fugue_reference["output"],
        _fugue_evidence,
    )
    if not isinstance(_fugue_result, dict) or not _fugue_result:
        raise ValueError("score() must return a non-empty object")
    _fugue_values = []
    for _fugue_name, _fugue_value in _fugue_result.items():
        if not isinstance(_fugue_name, str) or not _fugue_name:
            raise ValueError("score dimension names must be non-empty strings")
        if isinstance(_fugue_value, bool):
            _fugue_values.append(1.0 if _fugue_value else 0.0)
        elif (
            isinstance(_fugue_value, (int, float))
            and not isinstance(_fugue_value, bool)
            and math.isfinite(float(_fugue_value))
            and 0 <= float(_fugue_value) <= 1
        ):
            _fugue_values.append(float(_fugue_value))
        else:
            raise ValueError("score dimensions must be bool or numbers in 0..1")
    print(json.dumps({
        "score": min(_fugue_values),
        "reason": "custom deterministic scorer",
        "details": _fugue_result,
    }, sort_keys=True))
"""
    )
    profiles = load_task_profiles(repo_root)
    profile = profiles.scorer_runtime(evaluator.runtime)
    limits = TaskAuthoringLimitsV1(
        max_tasks=1,
        max_scenarios=1,
        max_prompt_bytes=1,
        max_authored_asset_bytes=1,
        max_user_turns=1,
        max_agent_turns=1,
        max_interactor_calls=0,
        max_judge_calls=0,
        scorer_timeout_sec=30,
        scorer_memory_mb=256,
        scorer_cpus=1.0,
        scorer_output_bytes=64_000,
    )
    return run_inline_scorer(
        source=wrapper,
        evidence=dict(evidence),
        reference={
            "task": dict(task),
            "output": output,
            "expected": expected,
        },
        profile=profile,
        limits=limits,
    )


def _validate_custom_scorer_source(source: str) -> None:
    if len(source.encode()) > 32_000:
        raise ValueError("custom scorer source exceeds 32,000 bytes")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("custom scorer is not valid Python") from exc
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "score"
    ]
    if len(definitions) != 1:
        raise ValueError("custom scorer must define exactly one score function")
    function = definitions[0]
    if (
        len(function.args.args) != 3
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
        or function.args.defaults
    ):
        raise ValueError("score must have the signature score(task, output, evidence)")
    if [argument.arg for argument in function.args.args] != [
        "task",
        "output",
        "evidence",
    ]:
        raise ValueError("score must have the signature score(task, output, evidence)")


def _custom_scorer_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    permitted = (
        "artifacts",
        "artifact_paths",
        "changed_paths",
        "inspected_paths",
        "opened_paths",
        "retrieved_paths",
        "tool_calls",
        "trace_summary",
    )
    return {
        key: row[key]
        for key in permitted
        if key in row
    }


def _contains_expected(output: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(output, dict) and all(
            key in output and _contains_expected(output[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(output, list) and all(
            any(_contains_expected(candidate, item) for candidate in output)
            for item in expected
        )
    return output == expected


def _is_json(value: str) -> bool:
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return False
    return True


def _judge_calibration_issue(
    judge: ComparisonEvaluatorV1, repo_root: Path
) -> str | None:
    if not judge.calibration:
        return f"judge {judge.id} has no reviewed calibration result"
    path = _safe_input_path(Path(judge.calibration), repo_root, "judge calibration")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return f"judge {judge.id} calibration must be a mapping"
    if value.get("schema_version") != 1:
        return f"judge {judge.id} calibration schema is unsupported"
    if value.get("review_status") != "adjudicated":
        return f"judge {judge.id} calibration is not adjudicated"
    if int(value.get("reviewers_per_example") or 0) < 2:
        return f"judge {judge.id} calibration was not double-reviewed"
    if value.get("disagreements_adjudicated") is not True:
        return f"judge {judge.id} calibration disagreements are unresolved"
    if value.get("judge_profile") != judge.profile:
        return f"judge {judge.id} calibration profile does not match"
    if value.get("rubric_digest") != _judge_contract_digest(judge):
        return f"judge {judge.id} calibration rubric does not match"
    examples = int(value.get("examples") or 0)
    true_positive = float(value.get("true_positive_rate") or 0)
    true_negative = float(value.get("true_negative_rate") or 0)
    critical_false_passes = int(value.get("critical_false_passes") or 0)
    if examples < 48:
        return f"judge {judge.id} calibration has fewer than 48 examples"
    if true_positive < 0.85 or true_negative < 0.85:
        return f"judge {judge.id} calibration is below 0.85 TPR/TNR"
    if critical_false_passes:
        return f"judge {judge.id} has critical false passes"
    return None


def _judge_contract_digest(judge: ComparisonEvaluatorV1) -> str:
    return stable_digest(
        {
            "schema_version": 1,
            "judge_id": judge.id,
            "profile": judge.profile,
            "rubric": judge.rubric,
            "dimensions": list(judge.dimensions),
            "evidence": list(judge.evidence),
        }
    )


def _behavior_diff(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], prefix: str = ""
) -> set[str]:
    result: set[str] = set()
    for key in sorted(set(baseline) | set(candidate)):
        path = f"{prefix}.{key}" if prefix else str(key)
        left = baseline.get(key)
        right = candidate.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            result.update(_behavior_diff(left, right, path))
        elif left != right:
            result.add(path)
    return result


def _preview_dict(value: PreviewSummary) -> dict[str, Any]:
    return {
        "cells": value.cells,
        "applicable_cells": value.applicable_cells,
        "estimated_trials": value.estimated_trials,
        "harnesses": list(value.harnesses),
        "variants": list(value.variants),
        "systems": list(value.systems),
        "workloads": list(value.workloads),
        "matrix_cells": [asdict(item) for item in value.matrix_cells],
    }


def _result_markdown(result: ComparisonResultV1) -> str:
    mechanism = "".join(
        (
            f"- {stage.replace('_', ' ').title()}: "
            f"baseline {values['baseline']['observed']}/"
            f"{values['baseline']['applicable']}; "
            f"candidate {values['candidate']['observed']}/"
            f"{values['candidate']['applicable']}\n"
        )
        for stage, values in result.mechanism_summary.items()
    )
    judge = (
        "No blind judge was used.\n"
        if result.judge_summary.get("status") == "not_used"
        else "Blind-judge dimensions are available in `result.json`.\n"
    )
    return (
        f"# {result.comparison_id}\n\n"
        f"- Rows: {result.rows}\n"
        f"- Baseline passed: {result.baseline_passed}\n"
        f"- Candidate passed: {result.candidate_passed}\n"
        f"- Improved pairs: {result.improved}\n"
        f"- Regressed pairs: {result.regressed}\n"
        f"- Unchanged pairs: {result.unchanged}\n"
        f"- Incomplete pairs: {result.incomplete}\n\n"
        f"- Required evaluations incomplete: "
        f"{result.required_evaluations_incomplete}\n\n"
        "## Operational health\n\n"
        f"- Infrastructure failures: "
        f"{result.operational_summary['infrastructure_failures']}\n"
        f"- Execution states: "
        f"`{json.dumps(result.operational_summary['execution_states'], sort_keys=True)}`\n"
        f"- Evidence states: "
        f"`{json.dumps(result.operational_summary['evidence_states'], sort_keys=True)}`\n"
        f"- Observed cost: "
        f"{result.operational_summary['observed_cost_usd'] if result.operational_summary['observed_cost_usd'] is not None else 'unavailable'}\n\n"
        "## Mechanism evidence\n\n"
        + (mechanism or "No mechanism evidence was available.\n")
        + "\n## Blind judge\n\n"
        + judge
        + "\n"
        "## Limitations\n\n"
        + "".join(f"- {item}\n" for item in result.limitations)
    )


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} must contain at least one row")
    return rows


def _portable_input_path(
    value: Any, source: Path, repo_root: Path, label: str
) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} path is required")
    path = Path(text)
    resolved = (source / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    if ".." in relative.parts:
        raise ValueError(f"{label} path is unsafe")
    return relative.as_posix()


def _safe_input_path(path: Path, repo_root: Path, label: str) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _artifact_digest(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned[field] = ""
    return stable_digest(unsigned)


def _verify_artifact(value: Mapping[str, Any], field: str, label: str) -> None:
    supplied = str(value.get(field) or "")
    if len(supplied) != 64 or _artifact_digest(value, field) != supplied:
        raise ValueError(f"{label} digest does not match")


def _sha256_path(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _schema(raw: Mapping[str, Any], label: str) -> int:
    version = raw.get("schema_version")
    if version != COMPARISON_SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema_version must be {COMPARISON_SCHEMA_VERSION}"
        )
    return COMPARISON_SCHEMA_VERSION


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _sequence(
    value: Any, label: str, *, allow_empty: bool = False
) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return value


def _string_tuple(
    value: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    values = _sequence(value, label, allow_empty=allow_empty)
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"{label} values must be non-empty strings")
    return tuple(str(item).strip() for item in values)


def _text(value: Any, label: str, limit: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    if len(text.encode()) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    return text


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a non-negative number")
    return result


def _reject_unknown(
    value: Mapping[str, Any], allowed: set[str] | frozenset[str], label: str
) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")


def _drop_empty(
    value: dict[str, Any], *, preserve_false: bool = False
) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item not in (None, "", [], (), {})
        and (preserve_false or item is not False)
    }
