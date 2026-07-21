from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml

from fugue.bench.context import get_context_system
from fugue.bench.files import require_unique
from fugue.bench.integrations import declared_mcp_servers, effective_selections
from fugue.bench.library import (
    BuiltinScorerSelection,
    EvaluationSourceSpec,
    ExperimentSpec,
    RubricScorerSelection,
    WorkloadSpec,
    get_prompt,
    get_skill,
    validate_id,
)
from fugue.bench.manifest import BenchmarkManifest
from fugue.model_plane import (
    ModelRoute,
    provider_request_headers,
    resolve_model_route,
)
from fugue.redaction import redact_value

EVALUATIONS_ROOT = Path("configs/fugue/evaluations")
GENERATED_DATASET_ROOT = Path(".fugue/cache/datasets/generated")
CASE_FILE = "cases.jsonl"
RUBRIC_FILE = "rubric.yaml"
MANIFEST_FILE = "manifest.yaml"
EVALUATION_DIMENSIONS = {
    "task_completion",
    "correctness",
    "groundedness",
    "tool_use",
    "artifact_quality",
}
CASE_STRATA = {"easy", "boundary", "failure", "integration"}
DEFAULT_JUDGE_THRESHOLD = 0.7
MAX_SOURCE_CHARS = 64_000
MAX_SOURCE_TOTAL_CHARS = 512_000
MAX_MCP_ITEMS = 50
MAX_MCP_DISCOVERY_SECONDS = 30
MAX_GENERATED_CASE_BYTES = 12_000
MAX_GENERATED_RUBRIC_BYTES = 12_000

EvaluationAssetKind = Literal[
    "evaluation_cases",
    "evaluation_rubric",
    "evaluation_manifest",
]


@dataclass(frozen=True)
class EvaluationFile:
    kind: EvaluationAssetKind
    suite_id: str
    path: Path
    body: str
    sha256: str


@dataclass(frozen=True)
class EvaluationDraft:
    suite_id: str
    cases: tuple[dict[str, Any], ...]
    rubric: dict[str, Any]
    files: tuple[EvaluationFile, ...]
    coverage: dict[str, int]

    @property
    def overlay(self) -> dict[str, str]:
        return {item.path.as_posix(): item.body for item in self.files}


@dataclass(frozen=True)
class EvaluationSource:
    id: str
    kind: str
    title: str
    sha256: str
    content: str
    metadata: dict[str, Any]

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "sha256": self.sha256,
            "content": redact_value(self.content),
            "metadata": redact_value(self.metadata),
        }


def evaluation_asset_path(kind: EvaluationAssetKind, suite_id: str) -> Path:
    validate_id(suite_id, kind="evaluation suite id")
    filename = {
        "evaluation_cases": CASE_FILE,
        "evaluation_rubric": RUBRIC_FILE,
        "evaluation_manifest": MANIFEST_FILE,
    }[kind]
    return EVALUATIONS_ROOT / suite_id / filename


def build_evaluation_draft(
    raw: Any,
    experiment: ExperimentSpec,
    *,
    generator_model: str,
    source_catalog: Sequence[EvaluationSource],
    repo_root: Path | None = None,
) -> tuple[ExperimentSpec, EvaluationDraft]:
    if not isinstance(raw, dict):
        raise ValueError("evaluation draft must be an object")
    unknown = sorted(set(raw) - {"suite_id", "cases", "rubric"})
    if unknown:
        raise ValueError(f"unknown evaluation draft field(s): {', '.join(unknown)}")
    configured = experiment.evaluation_generation
    suite_id = validate_id(
        configured.suite_id
        if configured is not None
        else str(raw.get("suite_id") or f"{experiment.id}-capability-smoke"),
        kind="evaluation suite id",
    )
    if raw.get("suite_id") and str(raw["suite_id"]) != suite_id:
        raise ValueError(
            f"evaluation suite id must be the configured value {suite_id!r}"
        )
    _validate_capability_baselines(experiment)
    expected_size = (
        experiment.evaluation_generation.size
        if experiment.evaluation_generation is not None
        else 8
    )
    catalog = {item.id: item for item in source_catalog}
    proposed = tuple(
        _evaluation_case(item, index, catalog)
        for index, item in enumerate(raw.get("cases") or [], start=1)
    )
    require_unique([str(item["id"]) for item in proposed], "evaluation case")
    del repo_root
    cases = proposed
    if len(cases) != expected_size:
        raise ValueError(
            f"evaluation suite {suite_id} requires exactly {expected_size} cases; "
            f"received {len(cases)}"
        )
    if expected_size >= 8:
        strata = {
            tag
            for case in cases
            for tag in case.get("tags") or []
            if tag in CASE_STRATA
        }
        missing_strata = sorted(CASE_STRATA - strata)
        if missing_strata:
            raise ValueError(
                "evaluation suite is missing case strata: " + ", ".join(missing_strata)
            )
    require_unique([str(item["id"]) for item in cases], "evaluation case")
    rubric = _evaluation_rubric(
        raw.get("rubric"),
        suite_id=suite_id,
        cases=cases,
        generator_model=generator_model,
        source_catalog=source_catalog,
    )
    case_body = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in cases
    )
    rubric_body = yaml.safe_dump(rubric, sort_keys=False)
    case_sha = _sha256(case_body)
    rubric_sha = _sha256(rubric_body)
    fingerprint = _stable_digest(
        {
            "cases_sha256": case_sha,
            "rubric_sha256": rubric_sha,
            "suite_id": suite_id,
        }
    )
    manifest = _generated_manifest(
        experiment,
        suite_id=suite_id,
        cases=cases,
        cases_sha256=case_sha,
        rubric_sha256=rubric_sha,
        fingerprint=fingerprint,
    )
    manifest_body = yaml.safe_dump(manifest, sort_keys=False)
    files = tuple(
        EvaluationFile(
            kind=kind,
            suite_id=suite_id,
            path=evaluation_asset_path(kind, suite_id),
            body=body,
            sha256=_sha256(body),
        )
        for kind, body in (
            ("evaluation_cases", case_body),
            ("evaluation_rubric", rubric_body),
            ("evaluation_manifest", manifest_body),
        )
    )
    updated = attach_evaluation_suite(
        experiment,
        suite_id,
        workload_id=(
            configured.workload_id if configured is not None else "capabilities"
        ),
    )
    coverage: dict[str, int] = {}
    for case in cases:
        family = str(case["family"])
        coverage[family] = coverage.get(family, 0) + 1
    return updated, EvaluationDraft(
        suite_id=suite_id,
        cases=cases,
        rubric=rubric,
        files=files,
        coverage=coverage,
    )


def attach_evaluation_suite(
    experiment: ExperimentSpec, suite_id: str, *, workload_id: str
) -> ExperimentSpec:
    manifest_path = evaluation_asset_path("evaluation_manifest", suite_id)
    rubric_path = evaluation_asset_path("evaluation_rubric", suite_id).as_posix()
    scorer_refs = [
        BuiltinScorerSelection(type="builtin", id="harbor-outcome"),
        RubricScorerSelection(type="rubric", path=rubric_path),
    ]
    workloads = list(experiment.workloads)
    target_index = next(
        (
            index
            for index, workload in enumerate(workloads)
            if workload.id == workload_id
        ),
        None,
    )
    if target_index is None:
        workloads.append(
            WorkloadSpec(
                id=workload_id,
                runner="harbor",
                manifest=manifest_path,
                scorers=scorer_refs,
            )
        )
    else:
        current = workloads[target_index]
        if current.runner != "harbor":
            raise ValueError(
                f"evaluation workload {workload_id} already exists with runner "
                f"{current.runner!r}"
            )
        if current.manifest not in {None, manifest_path}:
            raise ValueError(
                f"evaluation workload {workload_id} already targets another manifest"
            )
        workloads[target_index] = replace(
            current,
            manifest=manifest_path,
            scorers=scorer_refs,
        )

    presets = [
        replace(
            preset,
            workloads=(
                list(preset.workloads)
                if workload_id in preset.workloads
                else [*preset.workloads, workload_id]
            ),
        )
        if preset.workloads
        else preset
        for preset in experiment.presets
    ]
    return replace(experiment, workloads=workloads, presets=presets)


def _validate_capability_baselines(experiment: ExperimentSpec) -> None:
    variants = [variant for variant in experiment.variants if variant.enabled]
    if not variants:
        raise ValueError("generated evaluations require at least one enabled variant")
    skill_ids = {skill_id for variant in variants for skill_id in variant.skill_ids}
    for skill_id in sorted(skill_ids):
        if all(skill_id in variant.skill_ids for variant in variants):
            raise ValueError(
                f"generated evaluation requires a baseline that omits skill {skill_id}"
            )
    feature_sets = []
    for variant in variants:
        features = {
            *(f"skill:{value}" for value in variant.skills),
            *(f"integration:{value.id}" for value in variant.integrations),
        }
        if variant.prompt_id:
            features.add(f"prompt:{variant.prompt_id}")
        if variant.context.system_id != "none":
            features.add(
                f"context:{variant.context.system_id}:{variant.context.delivery}"
            )
        feature_sets.append(features)
    for feature in sorted(set().union(*feature_sets)):
        if all(feature in values for values in feature_sets):
            raise ValueError(
                "generated evaluation requires a baseline that omits " + feature
            )


def needs_evaluation_generation(experiment: ExperimentSpec) -> bool:
    if experiment.evaluation_generation is not None and not any(
        workload.scorers for workload in experiment.workloads
    ):
        return True
    if any(
        workload.runner == "harbor" and workload.manifest is None
        for workload in experiment.workloads
    ):
        return True
    if not experiment.workloads and not str(experiment.manifest):
        return True
    return False


def source_catalog(
    experiment: ExperimentSpec,
    repo_root: Path,
    *,
    allow_mcp_io: bool = False,
    draft_assets: Mapping[tuple[str, str], str] | None = None,
) -> tuple[EvaluationSource, ...]:
    values: list[EvaluationSource] = []
    remaining = MAX_SOURCE_TOTAL_CHARS
    configured = (
        experiment.evaluation_generation.sources
        if experiment.evaluation_generation is not None
        else []
    )
    for index, source in enumerate(configured, start=1):
        discovered = _configured_source(
            source,
            index=index,
            experiment=experiment,
            repo_root=repo_root,
            allow_mcp_io=allow_mcp_io,
        )
        for item in discovered:
            if remaining <= 0:
                break
            selected = item.content[: min(MAX_SOURCE_CHARS, remaining)]
            remaining -= len(selected)
            values.append(replace(item, content=selected))

    asset_bodies = draft_assets or {}
    for variant in experiment.variants:
        if variant.prompt_id:
            body = asset_bodies.get(("prompt", variant.prompt_id))
            if body is not None:
                values.append(
                    _source(
                        f"prompt:{variant.prompt_id}",
                        "prompt",
                        variant.prompt_id,
                        body,
                        {"draft": True},
                    )
                )
            else:
                prompt = get_prompt(variant.prompt_id, repo_root)
                values.append(
                    _source(
                        f"prompt:{prompt.id}",
                        "prompt",
                        prompt.title,
                        prompt.body[:MAX_SOURCE_CHARS],
                        {"path": prompt.path},
                    )
                )
        for skill_id in variant.skill_ids:
            body = asset_bodies.get(("skill", skill_id))
            if body is not None:
                values.append(
                    _source(
                        f"skill:{skill_id}",
                        "skill",
                        skill_id,
                        body,
                        {"draft": True},
                    )
                )
            else:
                skill = get_skill(skill_id, repo_root)
                values.append(
                    _source(
                        f"skill:{skill.id}",
                        "skill",
                        skill.title,
                        skill.body[:MAX_SOURCE_CHARS],
                        {"path": skill.path},
                    )
                )
    for server in _declared_mcp_servers(experiment, repo_root):
        public = _public_mcp_definition(server)
        if public:
            name = str(public.get("name") or public.get("id") or "mcp")
            values.append(
                _source(
                    f"mcp:{name}",
                    "mcp",
                    name,
                    json.dumps(public, sort_keys=True),
                    {"discovery": "configuration"},
                )
            )
    bounded: list[EvaluationSource] = []
    remaining = MAX_SOURCE_TOTAL_CHARS
    for item in _dedupe_sources(values):
        if remaining <= 0:
            break
        content = item.content[: min(MAX_SOURCE_CHARS, remaining)]
        remaining -= len(content)
        bounded.append(replace(item, content=content))
    return tuple(bounded)


def load_cases(path: Path, *, text: str | None = None) -> tuple[dict[str, Any], ...]:
    raw = text if text is not None else path.read_text()
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: evaluation case must be an object")
        _validate_saved_case(value, path, number)
        rows.append(value)
    require_unique([str(item.get("id") or "") for item in rows], "evaluation case")
    return tuple(rows)


def load_rubric(path: Path, *, text: str | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(text if text is not None else path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: evaluation rubric must be a mapping")
    _validate_saved_rubric(raw, path)
    return raw


def scorer_bundle(
    refs: Sequence[str],
    repo_root: Path,
    *,
    overlay: Mapping[str, str] | None = None,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...], dict[str, str]]:
    builtins: list[str] = []
    rubrics: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    assets = overlay or {}
    for ref in refs:
        if ref.startswith("builtin:"):
            builtins.append(ref)
            hashes[ref] = _sha256(ref)
            continue
        path = Path(ref)
        resolved = path if path.is_absolute() else repo_root / path
        text = assets.get(path.as_posix())
        rubric = load_rubric(resolved, text=text)
        rubrics.append(rubric)
        hashes[ref] = _sha256(text if text is not None else resolved.read_text())
    return tuple(builtins), tuple(rubrics), hashes


def apply_generated_evaluation(
    row: dict[str, Any],
    *,
    case: Mapping[str, Any],
    rubrics: Sequence[Mapping[str, Any]],
    judge_model: str,
    env: Mapping[str, str],
    trial_dir: Path,
    judge_request: Any | None = None,
) -> None:
    """Add separate generated-evaluation dimensions without altering Harbor status."""
    answer = _trial_answer(trial_dir, row)
    deterministic = _deterministic_assertions(
        case,
        row=row,
        answer=answer,
        trial_dir=trial_dir,
    )
    row["evaluation_assertions"] = deterministic
    if not rubrics:
        row["evaluation_judge_status"] = "not_requested"
        return
    if not judge_model:
        row["evaluation_judge_status"] = "failed"
        row["evaluation_error"] = (
            "ValueError: generated evaluation scoring requires an explicit judge_model"
        )
        return
    dimensions = list(dict.fromkeys(str(v) for v in case["scorer_dimensions"]))
    definitions = {
        str(value["id"]): value
        for rubric in rubrics
        for value in rubric.get("dimensions") or []
        if isinstance(value, dict)
    }
    missing = sorted(set(dimensions) - set(definitions))
    if missing:
        row["evaluation_judge_status"] = "failed"
        row["evaluation_error"] = (
            f"ValueError: rubric is missing dimension(s): {', '.join(missing)}"
        )
        return
    evidence = {
        "answer": answer,
        "artifact_paths": sorted(
            str(path.relative_to(trial_dir))
            for path in trial_dir.rglob("*")
            if path.is_file() and "artifact" in path.as_posix().lower()
        )[:100]
        if trial_dir.is_dir()
        else [],
        "observed_tools": row.get("weave_tool_names"),
        "harbor_status": row.get("status"),
        "harbor_pass": row.get("pass"),
    }
    request = judge_request or _generated_judge_request
    started = time.perf_counter()
    try:
        payload, usage = request(
            model=judge_model,
            env=env,
            case=redact_value(dict(case)),
            dimensions=[dict(definitions[value]) for value in dimensions],
            evidence=redact_value(evidence),
            deterministic=redact_value(deterministic),
        )
        raw_scores = payload.get("scores") or {}
        if not isinstance(raw_scores, dict):
            raise ValueError("judge scores must be an object")
        reasons = payload.get("reasons") or {}
        if not isinstance(reasons, dict):
            raise ValueError("judge reasons must be an object")
        na: list[str] = []
        for dimension in dimensions:
            if dimension in deterministic and deterministic[dimension] is None:
                na.append(dimension)
                continue
            value = raw_scores.get(dimension)
            if value is None:
                na.append(dimension)
                continue
            score = float(value)
            if not 0 <= score <= 1:
                raise ValueError(f"judge {dimension} must be between 0 and 1")
            deterministic_score = deterministic.get(dimension)
            if isinstance(deterministic_score, (int, float)):
                score = min(score, float(deterministic_score))
            row[f"evaluation_{dimension}"] = score
        row.update(
            {
                "evaluation_na_dimensions": na,
                "evaluation_judge_model": judge_model,
                "evaluation_judge_reasons": {
                    key: str(value)[:1_000] for key, value in reasons.items()
                },
                "evaluation_judge_input_tokens": usage.get("input_tokens"),
                "evaluation_judge_output_tokens": usage.get("output_tokens"),
            }
        )
        row["evaluation_judge_reasons"] = redact_value(row["evaluation_judge_reasons"])
        row["evaluation_judge_status"] = "scored"
    except Exception as exc:
        row["evaluation_judge_status"] = "failed"
        row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["evaluation_judge_latency_ms"] = (time.perf_counter() - started) * 1000


def _deterministic_assertions(
    case: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    answer: str | None,
    trial_dir: Path,
) -> dict[str, Any]:
    expected = case.get("expected") or {}
    result: dict[str, Any] = {
        "task_completion": float(
            row.get("pass") is True or row.get("status") == "passed"
        )
    }
    facts = [str(value) for value in expected.get("facts") or []]
    if facts:
        normalized = (answer or "").casefold()
        matched = [value.casefold() in normalized for value in facts]
        result["correctness"] = sum(matched) / len(matched)
        result["fact_assertions"] = dict(zip(facts, matched, strict=True))
    artifacts = list(expected.get("artifacts") or [])
    if artifacts:
        checks: dict[str, bool] = {}
        for assertion in artifacts:
            requested = str(assertion["path"])
            candidates = list(trial_dir.rglob(Path(requested).name))
            path = candidates[0] if candidates else None
            for check in assertion.get("checks") or []:
                key = f"{requested}:{check}"
                if check == "exists":
                    checks[key] = path is not None and path.is_file()
                elif check == "nonempty":
                    checks[key] = bool(path and path.is_file() and path.stat().st_size)
                elif check == "json":
                    try:
                        if path is None:
                            raise ValueError("missing")
                        json.loads(path.read_text())
                    except (OSError, ValueError, json.JSONDecodeError):
                        checks[key] = False
                    else:
                        checks[key] = True
        result["artifact_quality"] = (
            sum(checks.values()) / len(checks) if checks else None
        )
        result["artifact_assertions"] = checks
    tool_calls = list(expected.get("tool_calls") or [])
    if tool_calls:
        observed = row.get("weave_tool_names")
        if isinstance(observed, dict):
            matched = {
                str(assertion["tool"]): bool(observed.get(str(assertion["tool"])))
                for assertion in tool_calls
            }
            result["tool_use"] = sum(matched.values()) / len(matched)
            result["tool_assertions"] = matched
        else:
            result["tool_use"] = None
            result["tool_assertions"] = "N/A: tool telemetry unavailable"
    return result


def _trial_answer(trial_dir: Path, row: Mapping[str, Any]) -> str | None:
    value = row.get("agent_response")
    if isinstance(value, str) and value.strip():
        return value[:16_000]
    if not trial_dir.is_dir():
        return None
    candidates = list(trial_dir.rglob("fugue-answer.md"))
    if not candidates:
        return None
    value = candidates[0].read_text(errors="replace").strip()
    return value[:16_000] or None


def _generated_judge_request(
    *,
    model: str,
    env: Mapping[str, str],
    case: Mapping[str, Any],
    dimensions: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    deterministic: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    route = resolve_model_route(model, env)
    api_key = env.get(route.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{route.api_key_env} is required for evaluation judging")
    prompt = (
        "Evaluate one capability task. Return only JSON with a scores object and "
        "a reasons object keyed by the requested dimension ids. Each score must be "
        "0..1 or null when the evidence required to judge that dimension is absent. "
        "Do not create an overall or composite score. Deterministic failures are "
        "hard constraints.\n\n"
        + json.dumps(
            {
                "case": case,
                "dimensions": dimensions,
                "evidence": evidence,
                "deterministic_assertions": deterministic,
            },
            sort_keys=True,
            default=str,
        )[:48_000]
    )
    with httpx.Client(timeout=120) as client:
        return _post_judge(client, route, api_key, env, prompt)


def _post_judge(
    client: httpx.Client,
    route: ModelRoute,
    api_key: str,
    env: Mapping[str, str],
    prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if route.messages_base_url:
        response = client.post(
            f"{route.messages_base_url}/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": route.model_id,
                "max_tokens": 1_200,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        body = response.json()
        content = "".join(
            str(item.get("text") or "")
            for item in body.get("content", [])
            if isinstance(item, dict)
        )
        raw_usage = body.get("usage") or {}
        usage = {
            "input_tokens": raw_usage.get("input_tokens"),
            "output_tokens": raw_usage.get("output_tokens"),
        }
    else:
        response = client.post(
            f"{route.chat_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                **provider_request_headers(route, env),
            },
            json={
                "model": route.model_id,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        body = response.json()
        content = str(
            ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        )
        raw_usage = body.get("usage") or {}
        usage = {
            "input_tokens": raw_usage.get("prompt_tokens"),
            "output_tokens": raw_usage.get("completion_tokens"),
        }
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError("judge returned no JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")
    return payload, usage


class GeneratedCapabilityMaterializer:
    def materialize(
        self,
        manifest: BenchmarkManifest,
        destination: Path,
        source_path: Path,
        *,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        rows = load_cases(source_path)
        if len(rows) != len(manifest.tasks):
            raise ValueError("generated manifest task count does not match cases")
        root = repo_root or Path.cwd()
        for task in manifest.tasks:
            index = task.metadata.get("source_index")
            if not isinstance(index, int) or not 0 <= index < len(rows):
                raise ValueError(f"{task.id}: invalid generated source_index")
            case = rows[index]
            if str(case.get("id")) != task.id:
                raise ValueError(f"{task.id}: generated case identity drift")
            _write_generated_task(destination / task.id, case, root)
        suite_ids = {
            str(task.metadata.get("suite_id"))
            for task in manifest.tasks
            if task.metadata.get("suite_id")
        }
        return {
            "tasks": len(rows),
            "suite_id": next(iter(suite_ids)) if len(suite_ids) == 1 else None,
        }


def _evaluation_case(
    raw: Any,
    index: int,
    source_catalog: Mapping[str, EvaluationSource],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("evaluation case must be an object")
    if (
        len(json.dumps(raw, separators=(",", ":"), default=str).encode())
        > MAX_GENERATED_CASE_BYTES
    ):
        raise ValueError(
            f"evaluation case exceeds {MAX_GENERATED_CASE_BYTES} serialized bytes"
        )
    allowed = {
        "id",
        "instruction",
        "family",
        "source_refs",
        "attachments",
        "expected",
        "scorer_dimensions",
        "tags",
        "turns",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown evaluation case field(s): {', '.join(unknown)}")
    case_id = validate_id(
        str(raw.get("id") or f"case-{index:03d}"), kind="evaluation case id"
    )
    instruction = str(raw.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"evaluation case {case_id} instruction is required")
    family = str(raw.get("family") or "agent").strip()
    if family not in {"prompt", "skill", "mcp", "agent", "mixed"}:
        raise ValueError(f"evaluation case {case_id} has unknown family {family}")
    refs = [str(value) for value in raw.get("source_refs") or []]
    if not refs:
        raise ValueError(f"evaluation case {case_id} requires source_refs")
    require_unique(refs, f"evaluation case {case_id} source")
    unknown_refs = sorted(set(refs) - set(source_catalog))
    if unknown_refs:
        raise ValueError(
            f"evaluation case {case_id} has unknown source ref(s): "
            f"{', '.join(unknown_refs)}"
        )
    expected = _expected_assertions(raw.get("expected"), case_id)
    attachments = [
        _attachment(value, case_id) for value in raw.get("attachments") or []
    ]
    default_dimensions = ["task_completion"]
    if expected["facts"] or expected.get("reference_answer"):
        default_dimensions.append("correctness")
    if refs:
        default_dimensions.append("groundedness")
    if expected["tool_calls"]:
        default_dimensions.append("tool_use")
    if expected["artifacts"]:
        default_dimensions.append("artifact_quality")
    dimensions = [
        str(value) for value in raw.get("scorer_dimensions") or default_dimensions
    ]
    unknown_dimensions = sorted(set(dimensions) - EVALUATION_DIMENSIONS)
    if unknown_dimensions:
        raise ValueError(
            f"evaluation case {case_id} has unknown scorer dimension(s): "
            f"{', '.join(unknown_dimensions)}"
        )
    turns = [
        str(value).strip() for value in raw.get("turns") or [] if str(value).strip()
    ]
    return {
        "schema_version": 1,
        "id": case_id,
        "instruction": instruction,
        "family": family,
        "source_refs": [
            {
                "id": ref,
                "kind": source_catalog[ref].kind,
                "sha256": source_catalog[ref].sha256,
            }
            for ref in refs
        ],
        "attachments": attachments,
        "expected": expected,
        "scorer_dimensions": list(dict.fromkeys(dimensions)),
        "tags": [str(value) for value in raw.get("tags") or []],
        **({"turns": turns} if turns else {}),
    }


def _expected_assertions(raw: Any, case_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"evaluation case {case_id} expected must be an object")
    unknown = sorted(
        set(raw) - {"facts", "tool_calls", "artifacts", "reference_answer"}
    )
    if unknown:
        raise ValueError(
            f"evaluation case {case_id} has unknown expected field(s): "
            f"{', '.join(unknown)}"
        )
    facts = [
        str(value).strip() for value in raw.get("facts") or [] if str(value).strip()
    ]
    tool_calls = [
        _tool_assertion(value, case_id) for value in raw.get("tool_calls") or []
    ]
    artifacts = [
        _artifact_assertion(value, case_id) for value in raw.get("artifacts") or []
    ]
    reference = str(raw.get("reference_answer") or "").strip() or None
    if not facts and not tool_calls and not artifacts:
        raise ValueError(
            f"evaluation case {case_id} needs a fact, tool, or artifact assertion"
        )
    return {
        "facts": facts,
        "tool_calls": tool_calls,
        "artifacts": artifacts,
        **({"reference_answer": reference} if reference else {}),
    }


def _tool_assertion(raw: Any, case_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"evaluation case {case_id} tool assertion must be an object")
    unknown = sorted(set(raw) - {"server", "tool", "arguments_subset"})
    if unknown:
        raise ValueError(
            f"evaluation case {case_id} has unknown tool assertion field(s): "
            f"{', '.join(unknown)}"
        )
    tool = str(raw.get("tool") or "").strip()
    if not tool:
        raise ValueError(f"evaluation case {case_id} tool name is required")
    server = str(raw.get("server") or "").strip() or None
    arguments = raw.get("arguments_subset") or {}
    if not isinstance(arguments, dict):
        raise ValueError(
            f"evaluation case {case_id} tool arguments_subset must be an object"
        )
    return {
        **({"server": server} if server else {}),
        "tool": tool,
        "arguments_subset": redact_value(arguments),
    }


def _artifact_assertion(raw: Any, case_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(
            f"evaluation case {case_id} artifact assertion must be an object"
        )
    unknown = sorted(set(raw) - {"path", "checks"})
    if unknown:
        raise ValueError(
            f"evaluation case {case_id} has unknown artifact assertion field(s): "
            f"{', '.join(unknown)}"
        )
    path = _safe_artifact_path(str(raw.get("path") or ""), case_id)
    checks = [str(value) for value in raw.get("checks") or ["exists", "nonempty"]]
    allowed = {"exists", "nonempty", "json"}
    unknown_checks = sorted(set(checks) - allowed)
    if unknown_checks:
        raise ValueError(
            f"evaluation case {case_id} has unknown artifact check(s): "
            f"{', '.join(unknown_checks)}"
        )
    return {"path": path, "checks": list(dict.fromkeys(checks))}


def _attachment(raw: Any, case_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"evaluation case {case_id} attachment must be an object")
    unknown = sorted(set(raw) - {"path", "target", "sha256"})
    if unknown:
        raise ValueError(
            f"evaluation case {case_id} has unknown attachment field(s): "
            f"{', '.join(unknown)}"
        )
    source = _safe_repo_path(str(raw.get("path") or ""), "attachment path")
    target = _safe_relative_path(
        str(raw.get("target") or source.name), "attachment target"
    )
    digest = str(raw.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"evaluation case {case_id} attachment sha256 is required")
    return {"path": source.as_posix(), "target": target.as_posix(), "sha256": digest}


def _evaluation_rubric(
    raw: Any,
    *,
    suite_id: str,
    cases: Sequence[dict[str, Any]],
    generator_model: str,
    source_catalog: Sequence[EvaluationSource],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("evaluation rubric must be an object")
    if (
        len(json.dumps(raw, separators=(",", ":"), default=str).encode())
        > MAX_GENERATED_RUBRIC_BYTES
    ):
        raise ValueError(
            f"evaluation rubric exceeds {MAX_GENERATED_RUBRIC_BYTES} serialized bytes"
        )
    unknown = sorted(set(raw) - {"dimensions"})
    if unknown:
        raise ValueError(f"unknown evaluation rubric field(s): {', '.join(unknown)}")
    dimensions = [_rubric_dimension(value) for value in raw.get("dimensions") or []]
    require_unique([str(value["id"]) for value in dimensions], "rubric dimension")
    dimension_ids = {str(value["id"]) for value in dimensions}
    required = {
        str(dimension) for case in cases for dimension in case["scorer_dimensions"]
    }
    missing = sorted(required - dimension_ids)
    if missing:
        raise ValueError(
            f"evaluation rubric is missing dimension(s): {', '.join(missing)}"
        )
    if "task_completion" not in dimension_ids:
        raise ValueError("evaluation rubric requires task_completion")
    source_hashes = {item.id: item.sha256 for item in source_catalog}
    return {
        "schema_version": 1,
        "id": suite_id,
        "dimensions": dimensions,
        "generation": {
            "generator_model": generator_model,
            "prompt_version": 1,
            "source_hashes": source_hashes,
        },
    }


def _rubric_dimension(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("rubric dimension must be an object")
    unknown = sorted(set(raw) - {"id", "kind", "criterion", "threshold", "evidence"})
    if unknown:
        raise ValueError(f"unknown rubric dimension field(s): {', '.join(unknown)}")
    dimension_id = str(raw.get("id") or "").strip()
    if dimension_id not in EVALUATION_DIMENSIONS:
        raise ValueError(f"unknown rubric dimension: {dimension_id}")
    kind = str(raw.get("kind") or "llm_judge")
    if kind != "llm_judge":
        raise ValueError(f"rubric dimension {dimension_id} kind must be llm_judge")
    criterion = str(raw.get("criterion") or "").strip()
    if not criterion:
        raise ValueError(f"rubric dimension {dimension_id} criterion is required")
    threshold = float(raw.get("threshold", DEFAULT_JUDGE_THRESHOLD))
    if not 0 <= threshold <= 1:
        raise ValueError(f"rubric dimension {dimension_id} threshold must be 0..1")
    evidence = [str(value) for value in raw.get("evidence") or []]
    return {
        "id": dimension_id,
        "kind": "llm_judge",
        "criterion": criterion,
        "threshold": threshold,
        "evidence": list(dict.fromkeys(evidence)),
    }


def _validate_saved_rubric(raw: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(raw) - {"schema_version", "id", "dimensions", "generation"})
    if unknown:
        raise ValueError(f"{path}: unknown rubric field(s): {', '.join(unknown)}")
    if int(raw.get("schema_version") or 0) != 1:
        raise ValueError(f"{path}: unsupported rubric schema_version")
    validate_id(str(raw.get("id") or ""), kind="evaluation rubric id")
    dimensions = raw.get("dimensions") or []
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError(f"{path}: rubric dimensions are required")
    parsed = [_rubric_dimension(value) for value in dimensions]
    require_unique([str(value["id"]) for value in parsed], "rubric dimension")


def _validate_saved_case(raw: dict[str, Any], path: Path, number: int) -> None:
    allowed = {
        "schema_version",
        "id",
        "instruction",
        "family",
        "source_refs",
        "attachments",
        "expected",
        "scorer_dimensions",
        "tags",
        "turns",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"{path}:{number}: unknown evaluation case field(s): " + ", ".join(unknown)
        )
    if int(raw.get("schema_version") or 0) != 1:
        raise ValueError(f"{path}:{number}: unsupported case schema_version")
    case_id = validate_id(
        str(raw.get("id") or ""),
        kind="evaluation case id",
    )
    if not str(raw.get("instruction") or "").strip():
        raise ValueError(f"{path}:{number}: case instruction is required")
    refs = raw.get("source_refs") or []
    if not isinstance(refs, list) or not refs:
        raise ValueError(f"{path}:{number}: case source_refs are required")
    source_ids: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError(f"{path}:{number}: source ref must be an object")
        source_id = str(ref.get("id") or "")
        if not source_id or not re.fullmatch(
            r"[0-9a-f]{64}", str(ref.get("sha256") or "")
        ):
            raise ValueError(f"{path}:{number}: source ref id and sha256 are required")
        source_ids.append(source_id)
    require_unique(source_ids, f"evaluation case {case_id} source")
    _expected_assertions(raw.get("expected"), case_id)
    for attachment in raw.get("attachments") or []:
        _attachment(attachment, case_id)
    dimensions = [str(value) for value in raw.get("scorer_dimensions") or []]
    if not dimensions or set(dimensions) - EVALUATION_DIMENSIONS:
        raise ValueError(f"{path}:{number}: invalid scorer_dimensions")


def _generated_manifest(
    experiment: ExperimentSpec,
    *,
    suite_id: str,
    cases: Sequence[dict[str, Any]],
    cases_sha256: str,
    rubric_sha256: str,
    fingerprint: str,
) -> dict[str, Any]:
    harness_names = experiment.harnesses or [
        "hermes",
        "openclaw",
        "claude-code",
        "codex",
    ]
    agents = {
        "hermes": "fugue.agents:FugueHermes",
        "openclaw": "fugue.agents:FugueOpenClaw",
        "claude-code": "fugue.agents:FugueClaudeCode",
        "codex": "fugue.agents:FugueCodex",
        "wba-responses": "fugue.agents:FugueWBAResponses",
    }
    unknown = sorted(set(harness_names) - set(agents))
    if unknown:
        raise ValueError(
            f"cannot generate Harbor manifest for harness(es): {', '.join(unknown)}"
        )
    return {
        "dataset": {
            "path": (GENERATED_DATASET_ROOT / fingerprint).as_posix(),
            "materializer": ("fugue.bench.evaluations:GeneratedCapabilityMaterializer"),
            "source": {
                "path": evaluation_asset_path("evaluation_cases", suite_id).as_posix(),
                "sha256": cases_sha256,
                "rubric": evaluation_asset_path(
                    "evaluation_rubric", suite_id
                ).as_posix(),
                "rubric_sha256": rubric_sha256,
            },
        },
        "model": experiment.model,
        "k": 1,
        "n_concurrent": experiment.n_concurrent or 2,
        "jobs_dir": f"jobs/{suite_id}",
        "harnesses": [{"name": name, "agent": agents[name]} for name in harness_names],
        "tasks": [
            {
                "id": str(case["id"]),
                "notes": str(case["instruction"])[:500],
                "metadata": {"source_index": index, "suite_id": suite_id},
            }
            for index, case in enumerate(cases)
        ],
    }


def _configured_source(
    source: EvaluationSourceSpec,
    *,
    index: int,
    experiment: ExperimentSpec,
    repo_root: Path,
    allow_mcp_io: bool,
) -> list[EvaluationSource]:
    if source.kind == "seed":
        assert source.text is not None
        return [
            _source(
                f"seed:{index}",
                "seed",
                f"Seed {index}",
                source.text,
                {},
            )
        ]
    if source.kind == "file":
        assert source.path is not None
        relative = _safe_repo_path(source.path, "evaluation source path")
        path = repo_root / relative
        if not path.resolve().is_relative_to(repo_root.resolve()):
            raise ValueError(f"evaluation source escapes repository: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"evaluation source not found: {relative}")
        content = path.read_text(errors="replace")
        return [
            _source(
                f"file:{relative.as_posix()}",
                "file",
                relative.name,
                content,
                {"path": relative.as_posix()},
            )
        ]
    assert source.kind == "mcp" and source.server is not None
    server = _mcp_server(experiment, source.server, repo_root)
    public = _public_mcp_definition(server)
    declaration = {
        "server": source.server,
        "tools": source.tools,
        "resources": source.resources,
        "configuration": public,
        "discovery": "declared",
    }
    declared = _source(
        f"mcp:{source.server}",
        "mcp",
        source.server,
        json.dumps(declaration, sort_keys=True),
        declaration,
    )
    if allow_mcp_io:
        discovered = _discover_mcp_source(source, server)
        if discovered:
            return [declared, *discovered]
    return [declared]


def _discover_mcp_source(
    source: EvaluationSourceSpec, server: dict[str, Any]
) -> list[EvaluationSource]:
    if not server.get("command"):
        return []
    try:
        import asyncio

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise RuntimeError(
            "MCP evaluation discovery requires the Fugue context extra"
        ) from exc

    async def discover() -> list[EvaluationSource]:
        params = StdioServerParameters(
            command=str(server["command"]),
            args=[str(value) for value in server.get("args") or []],
            env={str(k): str(v) for k, v in (server.get("env") or {}).items()},
        )
        values: list[EvaluationSource] = []
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools[:MAX_MCP_ITEMS]
                selected = set(source.tools)
                public_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in tools
                    if not selected or tool.name in selected
                ]
                missing_tools = sorted(
                    selected - {str(value["name"]) for value in public_tools}
                )
                if missing_tools:
                    raise ValueError(
                        f"MCP server {source.server} did not expose tool schema(s): "
                        + ", ".join(missing_tools)
                    )
                values.append(
                    _source(
                        f"mcp:{source.server}:tools",
                        "mcp_schema",
                        f"{source.server} tools",
                        json.dumps(public_tools, sort_keys=True, default=str),
                        {
                            "server": source.server,
                            "tools": [v["name"] for v in public_tools],
                        },
                    )
                )
                for uri in source.resources[:MAX_MCP_ITEMS]:
                    result = await session.read_resource(uri)
                    content = "\n".join(
                        str(getattr(item, "text", ""))
                        for item in result.contents
                        if getattr(item, "text", None)
                    )
                    values.append(
                        _source(
                            f"mcp:{source.server}:resource:{uri}",
                            "mcp_resource",
                            uri,
                            content,
                            {"server": source.server, "uri": uri},
                        )
                    )
        return values

    async def bounded() -> list[EvaluationSource]:
        return await asyncio.wait_for(
            discover(),
            timeout=MAX_MCP_DISCOVERY_SECONDS,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(bounded())
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, bounded()).result()


def _mcp_server(
    experiment: ExperimentSpec, server_id: str, repo_root: Path
) -> dict[str, Any]:
    values = _declared_mcp_servers(experiment, repo_root)
    for value in values:
        if str(value.get("name") or value.get("id") or "") == server_id:
            return value
    raise ValueError(f"unknown MCP server for evaluation source: {server_id}")


def _declared_mcp_servers(
    experiment: ExperimentSpec, repo_root: Path
) -> tuple[dict[str, Any], ...]:
    values: dict[str, dict[str, Any]] = {}
    for variant in experiment.variants:
        selections = effective_selections(
            experiment.integrations,
            variant.integrations,
        )
        servers = list(declared_mcp_servers(selections, repo_root))
        if variant.context.delivery == "native_mcp":
            spec = get_context_system(variant.context.system_id, repo_root)
            config = _merge_mapping(spec.config, variant.context.config)
            servers.extend(
                dict(item)
                for item in (config.get("binding") or {}).get("mcp_servers") or []
                if isinstance(item, dict)
            )
        for server in servers:
            name = str(server.get("name") or server.get("id") or "")
            if not name:
                continue
            previous = values.get(name)
            if previous is not None and previous != server:
                raise ValueError(f"MCP declaration {name!r} differs across variants")
            values[name] = server
    return tuple(values[name] for name in sorted(values))


def _merge_mapping(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(result.get(key), dict) and isinstance(value, Mapping):
            result[key] = _merge_mapping(result[key], value)
        else:
            result[key] = value
    return result


def _public_mcp_definition(value: Mapping[str, Any]) -> dict[str, Any]:
    secrets = tuple(
        str(item) for item in (value.get("env") or {}).values() if str(item).strip()
    )
    return {
        str(key): redact_value(item, secrets=secrets)
        for key, item in value.items()
        if str(key).lower() != "env"
        and not any(token in str(key).lower() for token in ("key", "secret", "token"))
    }


def _source(
    source_id: str,
    kind: str,
    title: str,
    content: str,
    metadata: dict[str, Any],
) -> EvaluationSource:
    selected = str(content)[:MAX_SOURCE_CHARS]
    return EvaluationSource(
        id=source_id,
        kind=kind,
        title=title,
        sha256=_sha256(selected),
        content=selected,
        metadata=metadata,
    )


def _dedupe_sources(values: Sequence[EvaluationSource]) -> list[EvaluationSource]:
    result: dict[str, EvaluationSource] = {}
    for value in values:
        result.setdefault(value.id, value)
    return list(result.values())


def _write_generated_task(root: Path, case: Mapping[str, Any], repo_root: Path) -> None:
    root.mkdir(parents=True)
    for name in ("environment", "solution", "tests"):
        (root / name).mkdir()
    attachments = list(case.get("attachments") or [])
    docker_lines = ["FROM python:3.12.10-slim-bookworm", "WORKDIR /workspace"]
    for index, item in enumerate(attachments):
        source = repo_root / _safe_repo_path(str(item["path"]), "attachment path")
        if not source.resolve().is_relative_to(repo_root.resolve()):
            raise ValueError(
                f"evaluation attachment escapes repository: {item['path']}"
            )
        if not source.is_file():
            raise FileNotFoundError(f"evaluation attachment not found: {item['path']}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"evaluation attachment checksum mismatch: {item['path']}")
        local = Path("fixtures") / f"{index:03d}-{source.name}"
        destination = root / "environment" / local
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        docker_lines.append(f"COPY {local.as_posix()} /workspace/{item['target']}")
    (root / "environment" / "Dockerfile").write_text("\n".join(docker_lines) + "\n")
    (root / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.3"',
                "",
                "[task]",
                f'name = "fugue/{case["id"]}"',
                'description = "Generated Fugue capability evaluation"',
                "",
                "[agent]",
                "timeout_sec = 900.0",
                "",
                "[verifier]",
                "timeout_sec = 60.0",
                "",
                "[environment]",
                "build_timeout_sec = 900.0",
                "cpus = 2",
                "memory_mb = 4096",
                "storage_mb = 10240",
                "",
            ]
        )
    )
    instruction = str(case["instruction"]).strip()
    turns = list(case.get("turns") or [])
    if turns:
        instruction += "\n\nConversation requirements:\n" + "\n".join(
            f"- {turn}" for turn in turns
        )
    (root / "instruction.md").write_text(
        f"# Capability task\n\n{instruction}\n\n"
        "Write the final answer to `/logs/artifacts/fugue-answer.md` and write "
        "`/logs/artifacts/fugue-evidence.json` as a JSON object with a `paths` "
        "array. Create any additional requested artifacts under `/logs/artifacts`.\n"
    )
    expected = dict(case["expected"])
    (root / "solution" / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    )
    (root / "solution" / "solve.sh").write_text(
        "#!/bin/sh\nmkdir -p /logs/artifacts\n"
        "printf '%s\n' 'Reference execution is intentionally unavailable.' "
        "> /logs/artifacts/fugue-answer.md\n"
        "printf '{\"paths\": []}\\n' > /logs/artifacts/fugue-evidence.json\n"
    )
    required = [str(item["path"]) for item in expected.get("artifacts") or []]
    (root / "tests" / "expected-artifacts.json").write_text(
        json.dumps(required, sort_keys=True) + "\n"
    )
    (root / "tests" / "test.sh").write_text(
        "#!/bin/sh\nmkdir -p /logs/verifier\npython - <<'PY'\n"
        "import json\nfrom pathlib import Path\n"
        "answer = Path('/logs/artifacts/fugue-answer.md')\n"
        "evidence = Path('/logs/artifacts/fugue-evidence.json')\n"
        "required = json.loads(Path('/tests/expected-artifacts.json').read_text())\n"
        "checks = {'answer_present': float(answer.is_file() and bool(answer.read_text().strip()))}\n"
        "try:\n"
        "    value = json.loads(evidence.read_text())\n"
        "    checks['evidence_format'] = float(isinstance(value, dict) and isinstance(value.get('paths'), list))\n"
        "except Exception:\n"
        "    checks['evidence_format'] = 0.0\n"
        "for item in required:\n"
        "    path = Path('/logs/artifacts') / item.removeprefix('/logs/artifacts/')\n"
        "    checks['artifact_' + path.name] = float(path.is_file() and path.stat().st_size > 0)\n"
        "Path('/logs/verifier/reward.json').write_text(json.dumps(checks, sort_keys=True))\n"
        "raise SystemExit(0 if all(checks.values()) else 1)\n"
        "PY\n"
    )
    for path in (root / "solution" / "solve.sh", root / "tests" / "test.sh"):
        path.chmod(0o755)


def _safe_repo_path(value: str, kind: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{kind} must be a repository-relative path")
    return path


def _safe_relative_path(value: str, kind: str) -> Path:
    path = _safe_repo_path(value, kind)
    if not path.parts:
        raise ValueError(f"{kind} is required")
    return path


def _safe_artifact_path(value: str, case_id: str) -> str:
    prefix = "/logs/artifacts/"
    if not value.startswith(prefix):
        raise ValueError(
            f"evaluation case {case_id} artifact path must start with {prefix}"
        )
    relative = _safe_relative_path(value.removeprefix(prefix), "artifact path")
    return f"{prefix}{relative.as_posix()}"


def _sha256(value: str | bytes) -> str:
    selected = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(selected).hexdigest()


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
