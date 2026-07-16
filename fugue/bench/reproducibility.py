from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fugue.bench.candidates import CANDIDATE_IDENTITY_SCHEMA_VERSION, stable_digest
from fugue.bench.context import (
    context_behavior_definition,
    context_behavior_digest,
    get_context_system,
)
from fugue.bench.files import atomic_write_json, store_consistent
from fugue.bench.library import ExperimentSpec, get_agent_preset, get_prompt
from fugue.bench.sources import resolve_skill

if TYPE_CHECKING:
    from fugue.bench.execution import PlannedCell
    from fugue.bench.job_config import RenderedJob

INPUT_LOCK_NAME = "input-lock.json"
EVALUATION_ASSET_LOCK_NAME = "evaluation-assets.json"
PREDICTION_ID_SCHEMA_VERSION = 1
_SENSITIVE_NAME = re.compile(
    r"(?:^|_)(?:api_?key|token|secret|password|credential|private_?key)(?:$|_)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RunSnapshotV1:
    schema_version: int
    identity_schema_version: int
    run_id: str
    experiment: dict[str, Any]
    request: dict[str, Any]
    assets: dict[str, dict[str, Any]]
    candidates: dict[str, dict[str, Any]]
    candidate_runtime: dict[str, dict[str, Any]]
    planned_matrix: tuple[dict[str, Any], ...]
    evaluation: dict[str, Any]
    runtime: dict[str, Any]
    required_env: tuple[str, ...]
    preset: dict[str, Any] | None = None
    source_experiment: dict[str, Any] | None = None
    resolved_experiment_sha256: str = ""
    capability_plan: tuple[dict[str, Any], ...] = ()
    planned_prediction_count: int = 0
    runtime_locks: tuple[dict[str, Any], ...] = ()
    publication_schema_version: int = 1
    evaluation_asset_lock_sha256: str = ""
    cohort_id: str = ""
    treatment_selection_sha256: str = ""
    snapshot_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["planned_matrix"] = list(self.planned_matrix)
        value["required_env"] = list(self.required_env)
        value["capability_plan"] = list(self.capability_plan)
        value["runtime_locks"] = list(self.runtime_locks)
        value["lock_sha256"] = self.snapshot_sha256
        return value


@dataclass(frozen=True)
class EvaluationAssetLockV1:
    schema_version: int
    run_id: str
    predictions: dict[str, dict[str, Any]]
    lock_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_snapshot(
    *,
    repo_root: Path,
    run_id: str,
    experiment: ExperimentSpec,
    request: Mapping[str, Any],
    jobs: list[RenderedJob],
    cells: list[PlannedCell],
    env: Mapping[str, str],
    evaluation_asset_lock_sha256: str = "",
    treatment_selection_sha256: str = "",
) -> RunSnapshotV1:
    secret_names = {
        value: name
        for name, value in env.items()
        if _SENSITIVE_NAME.search(name) and len(value) >= 8
    }
    required_env: set[str] = set()
    candidates: dict[str, dict[str, Any]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    executions: dict[str, dict[str, Any]] = {}
    assets: dict[str, dict[str, Any]] = {}
    generated_runtime_assets_by_config: dict[str, tuple[str, ...]] = {}
    fugue_source: dict[str, Any] | None = None
    for job in jobs:
        resolved = job.resolved_candidate
        store_consistent(
            candidates,
            job.candidate_id,
            resolved.definition,
            error=f"candidate {job.candidate_id} resolved inconsistently",
        )
        store_consistent(
            executions,
            resolved.execution_fingerprint,
            resolved.execution_definition,
            error=(
                f"execution {resolved.execution_fingerprint} resolved inconsistently"
            ),
        )
        selected_fugue_source = resolved.execution_definition.get("fugue_source")
        if selected_fugue_source is not None:
            if not isinstance(selected_fugue_source, dict):
                raise ValueError("Fugue source provenance must be an object")
            if fugue_source is not None and fugue_source != selected_fugue_source:
                raise ValueError("jobs resolved from different Fugue source states")
            fugue_source = selected_fugue_source
        generated_runtime_asset_ids: list[str] = []
        for runtime_file in job.generated_runtime_files:
            if not runtime_file.is_file():
                raise ValueError(f"generated runtime asset is missing: {runtime_file}")
            raw = runtime_file.read_bytes()
            try:
                body = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"generated runtime asset is not UTF-8: {runtime_file}"
                ) from exc
            runtime_path = _snapshot_path(runtime_file, repo_root)
            path_id = hashlib.sha256(runtime_path.encode()).hexdigest()[:12]
            asset_id = f"generated-runtime:{job.job_name}:{path_id}:{runtime_file.name}"
            record = {
                "kind": "generated_runtime",
                "path": runtime_path,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "body": body,
                "generated": True,
                "execution_fingerprint": resolved.execution_fingerprint,
            }
            store_consistent(
                assets,
                asset_id,
                record,
                error=f"generated runtime asset {asset_id} differs",
            )
            generated_runtime_asset_ids.append(asset_id)
        config_key = job.config_path.resolve().as_posix()
        selected_asset_ids = tuple(generated_runtime_asset_ids)
        store_consistent(
            generated_runtime_assets_by_config,
            config_key,
            selected_asset_ids,
            error=f"job config {job.config_path} has inconsistent assets",
        )
        context = get_context_system(job.context_system_id, repo_root)
        candidate_required_env: set[str] = {
            job.route.api_key_env,
            *context.required_env,
        }
        if job.prompt_id:
            prompt = get_prompt(job.prompt_id, repo_root)
            assets[f"prompt:{prompt.id}"] = {
                "kind": "prompt",
                "id": prompt.id,
                "path": prompt.path,
                "sha256": prompt.sha256,
                "body": prompt.body,
                "reviewed": True,
            }
        skill_assets: dict[str, dict[str, Any]] = {}
        for skill_id in job.skill_ids:
            skill = resolve_skill(skill_id, repo_root)
            skill_file = skill.path / "SKILL.md" if skill.path.is_dir() else skill.path
            record = {
                "kind": "skill",
                **skill.provenance(),
                "path": skill_file.as_posix(),
                "sha256": skill.digest.removeprefix("sha256:"),
                "body": skill_file.read_text(encoding="utf-8"),
                "reviewed": True,
            }
            assets[f"skill:{skill_id}"] = record
            skill_assets[skill_id] = record
        required_env.update(candidate_required_env)
        agent = _portable(
            dict((job.config.get("agents") or [{}])[0]),
            candidate_required_env,
            secret_names,
        )
        agent.pop("skills", None)
        environment = dict(job.config.get("environment") or {})
        environment.pop("mounts", None)
        environment.pop("extra_docker_compose", None)
        runtime = {
            "candidate_id": job.candidate_id,
            "harness": job.harness,
            "model_provider": job.route.provider,
            "model": job.route.display_model,
            "model_route": asdict(job.route),
            "context": {
                **context_behavior_definition(context),
                "serve_deliveries": sorted(context.serve_deliveries),
            },
            "context_config_hash": (job.config.get("fugue") or {}).get(
                "context_config_hash"
            ),
            "context_source_sha256": context_behavior_digest(context),
            "agent_config_hash": job.agent_config_hash,
            "content_hashes": (job.config.get("fugue") or {}).get("content_hashes")
            or {},
            "prompt_assets": {job.prompt_id: assets[f"prompt:{job.prompt_id}"]}
            if job.prompt_id
            else {},
            "skill_assets": skill_assets,
            "integration_ids": list(job.integration_ids),
            "agent": agent,
            "environment": _portable(environment, candidate_required_env, secret_names),
            "required_env": sorted(name for name in candidate_required_env if name),
        }
        context_runtime = resolved.execution_definition.get("context_runtime")
        if context_runtime is not None:
            runtime["context_runtime"] = context_runtime
        agent_runtime = resolved.execution_definition.get("agent_runtime")
        if agent_runtime is not None:
            runtime["agent_runtime"] = agent_runtime
        task_runtime = resolved.execution_definition.get("task_runtime")
        if task_runtime is not None:
            runtime["task_runtime"] = task_runtime
        if selected_fugue_source is not None:
            runtime["fugue_source"] = selected_fugue_source
        required_env.update(candidate_required_env)
        runtime["configuration_sha256"] = stable_digest(runtime)
        store_consistent(
            runtimes,
            job.candidate_id,
            runtime,
            error=f"candidate {job.candidate_id} runtime binding differs",
        )

    jobs_by_execution = {
        job.resolved_candidate.execution_fingerprint: job for job in jobs
    }
    planned_matrix = tuple(
        {
            "cell_id": cell.id,
            "candidate_id": cell.candidate_id,
            "execution_fingerprint": cell.execution_fingerprint,
            "execution_kind": cell.execution_kind,
            "comparison_example_id": cell.comparison_example_id,
            "trial_index": cell.trial_index,
            "workload_id": cell.workload_id,
            "task_id": cell.task_id,
            "applicable": cell.applicable,
            "skip_reason": cell.skip_reason,
            "config_path": cell.config_path.as_posix(),
            "config_sha256": cell.config_sha256,
            "result_path": cell.result_path.as_posix(),
            "runtime_assets": [list(item) for item in cell.runtime_assets],
            "generated_runtime_asset_ids": list(
                generated_runtime_assets_by_config.get(
                    cell.config_path.resolve().as_posix(),
                    (),
                )
            ),
            "planned_prediction_count": _planned_prediction_count(
                cell,
                jobs_by_execution.get(cell.execution_fingerprint),
            ),
        }
        for cell in cells
    )
    capability_plan = tuple(
        {
            "cell_id": cell.id,
            "workload_id": cell.workload_id,
            "execution_kind": cell.execution_kind,
            "harness": cell.harness,
            "context_system_id": cell.context_system_id,
            "delivery": cell.context_delivery,
            "applicable": cell.applicable,
            "reason": cell.skip_reason,
        }
        for cell in cells
    )
    scorer_hashes = {
        key: value for job in jobs for key, value in (job.scorer_hashes or {}).items()
    }
    evaluation = {
        "judge_model": experiment.judge_model,
        "generation": (
            asdict(experiment.evaluation_generation)
            if experiment.evaluation_generation is not None
            else None
        ),
        "scorer_hashes": scorer_hashes,
    }
    preset = None
    preset_id = str(request.get("agent_preset_id") or "")
    if preset_id:
        selected = get_agent_preset(preset_id, repo_root)
        preset = {
            "id": selected.id,
            "digest": stable_digest(selected.to_dict()),
        }
    resolved_experiment = _portable(experiment.to_dict(), required_env, secret_names)
    source_experiment = _source_experiment(experiment, repo_root)
    runtime_locks = tuple(
        sorted(
            (
                {
                    "candidate_id": candidate_id,
                    "configuration_sha256": runtime["configuration_sha256"],
                    "context_runtime": runtime.get("context_runtime"),
                    "agent_runtime": runtime.get("agent_runtime"),
                    "task_runtime": runtime.get("task_runtime"),
                }
                for candidate_id, runtime in runtimes.items()
            ),
            key=lambda item: item["candidate_id"],
        )
    )
    base = RunSnapshotV1(
        schema_version=1,
        identity_schema_version=CANDIDATE_IDENTITY_SCHEMA_VERSION,
        run_id=run_id,
        experiment=resolved_experiment,
        request=_portable(dict(request), required_env, secret_names),
        assets=assets,
        candidates=candidates,
        candidate_runtime=runtimes,
        planned_matrix=planned_matrix,
        evaluation=evaluation,
        runtime={
            "execution_fingerprints": sorted(
                {job.resolved_candidate.execution_fingerprint for job in jobs}
            ),
            "executions": executions,
            "fugue_source": fugue_source,
        },
        required_env=tuple(sorted(name for name in required_env if name)),
        preset=preset,
        source_experiment=source_experiment,
        resolved_experiment_sha256=stable_digest(resolved_experiment),
        capability_plan=capability_plan,
        planned_prediction_count=sum(
            int(item["planned_prediction_count"]) for item in planned_matrix
        ),
        runtime_locks=runtime_locks,
        evaluation_asset_lock_sha256=evaluation_asset_lock_sha256,
        cohort_id=str(request.get("cohort_id") or ""),
        treatment_selection_sha256=treatment_selection_sha256,
    )
    serialized = json.dumps(base.to_dict(), sort_keys=True, default=str)
    for name, value in env.items():
        if _SENSITIVE_NAME.search(name) and len(value) >= 8 and value in serialized:
            raise ValueError(f"refusing to serialize runtime secret: {name}")
    digest = stable_digest({**base.to_dict(), "lock_sha256": ""})
    return RunSnapshotV1(**{**asdict(base), "snapshot_sha256": digest})


def build_evaluation_asset_lock(
    run_id: str, cells: list[PlannedCell]
) -> EvaluationAssetLockV1:
    predictions = {
        _prediction_id(cell): {
            "task_id": cell.task_id,
            "expected_evidence_paths": list(cell.expected_evidence_paths),
        }
        for cell in cells
        if cell.expected_evidence_paths
    }
    base = EvaluationAssetLockV1(
        schema_version=1,
        run_id=run_id,
        predictions=predictions,
    )
    digest = stable_digest(base.to_dict())
    return EvaluationAssetLockV1(**{**asdict(base), "lock_sha256": digest})


def write_evaluation_asset_lock(repo_root: Path, lock: EvaluationAssetLockV1) -> Path:
    path = repo_root / ".fugue" / "runtime" / lock.run_id / EVALUATION_ASSET_LOCK_NAME
    payload = lock.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"evaluation asset lock already exists with different content: {path}"
            )
        path.chmod(0o600)
        return path
    return atomic_write_json(path, payload)


def read_evaluation_asset_lock(path: Path) -> EvaluationAssetLockV1:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("unsupported evaluation asset lock schema")
    expected = str(payload.get("lock_sha256") or "")
    unsigned = {**payload, "lock_sha256": ""}
    if not expected or stable_digest(unsigned) != expected:
        raise ValueError("evaluation asset lock digest does not match its content")
    return EvaluationAssetLockV1(
        schema_version=1,
        run_id=str(payload.get("run_id") or ""),
        predictions=dict(payload.get("predictions") or {}),
        lock_sha256=expected,
    )


def _prediction_id(cell: PlannedCell) -> str:
    return stable_digest(
        {
            "schema_version": PREDICTION_ID_SCHEMA_VERSION,
            "run_id": cell.run_id,
            "candidate_id": cell.candidate_id,
            "comparison_example_id": cell.comparison_example_id,
            "trial_index": cell.trial_index,
        }
    )


def write_run_input_lock(
    repo_root: Path,
    snapshot: RunSnapshotV1,
) -> Path:
    path = repo_root / ".fugue" / "runtime" / snapshot.run_id / INPUT_LOCK_NAME
    payload = snapshot.to_dict()
    _assert_no_secret_values(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"run input lock already exists with different content: {path}"
            )
        return path
    return atomic_write_json(path, payload)


def verify_snapshot(payload: Mapping[str, Any]) -> bool:
    if payload.get("schema_version") != 1:
        return False
    expected = str(payload.get("snapshot_sha256") or payload.get("lock_sha256") or "")
    unsigned = dict(payload)
    unsigned["snapshot_sha256"] = ""
    unsigned["lock_sha256"] = ""
    return bool(expected) and expected == stable_digest(unsigned)


def _planned_prediction_count(cell: PlannedCell, job: RenderedJob | None) -> int:
    if cell.execution_kind == "agent":
        return 1
    fugue = dict((job.config if job is not None else {}).get("fugue") or {})
    task_count = int(fugue.get("task_count") or 1)
    attempts = int(fugue.get("n_attempts") or cell.n_attempts or 1)
    return task_count * attempts


def _source_experiment(
    experiment: ExperimentSpec, repo_root: Path
) -> dict[str, Any] | None:
    path = repo_root / "configs" / "fugue" / "experiments" / f"{experiment.id}.yaml"
    if not path.is_file():
        return None
    return {
        "path": _snapshot_path(path, repo_root),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _portable(
    value: Any, required_env: set[str], secret_names: Mapping[str, str]
) -> Any:
    if isinstance(value, list | tuple):
        return [_portable(item, required_env, secret_names) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if (name == "env" or name.endswith("_env")) and isinstance(item, dict):
                mapped: dict[str, str] = {}
                for env_name, env_value in item.items():
                    env_name = str(env_name)
                    if _SENSITIVE_NAME.search(env_name):
                        required_env.add(env_name)
                        mapped[env_name] = f"${{{env_name}}}"
                    else:
                        mapped[env_name] = str(env_value)
                result[name] = mapped
            else:
                result[name] = _portable(item, required_env, secret_names)
        return result
    if isinstance(value, str) and value in secret_names:
        name = secret_names[value]
        required_env.add(name)
        return f"${{{name}}}"
    return value


def _snapshot_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _assert_no_secret_values(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            if _SENSITIVE_NAME.search(name) and isinstance(item, str):
                placeholder = bool(re.fullmatch(r"\$\{[A-Z][A-Z0-9_]*\}", item))
                env_name = name.endswith("_env") and bool(
                    re.fullmatch(r"[A-Z][A-Z0-9_]*", item)
                )
                if item and not placeholder and not env_name:
                    raise ValueError(
                        f"snapshot contains a credential-like value at {name}"
                    )
            _assert_no_secret_values(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _assert_no_secret_values(item)
