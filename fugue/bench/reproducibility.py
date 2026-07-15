from __future__ import annotations

import json
import os
import re
import uuid
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
from fugue.bench.library import ExperimentSpec, get_agent_preset, get_prompt
from fugue.bench.sources import resolve_skill

if TYPE_CHECKING:
    from fugue.bench.execution import PlannedCell
    from fugue.bench.job_config import RenderedJob

INPUT_LOCK_NAME = "input-lock.json"
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
    snapshot_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["planned_matrix"] = list(self.planned_matrix)
        value["required_env"] = list(self.required_env)
        value["lock_sha256"] = self.snapshot_sha256
        return value


def build_run_snapshot(
    *,
    repo_root: Path,
    run_id: str,
    experiment: ExperimentSpec,
    request: Mapping[str, Any],
    jobs: list[RenderedJob],
    cells: list[PlannedCell],
    env: Mapping[str, str],
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
    for job in jobs:
        resolved = job.resolved_candidate
        existing = candidates.get(job.candidate_id)
        if existing is not None and existing != resolved.definition:
            raise ValueError(f"candidate {job.candidate_id} resolved inconsistently")
        candidates[job.candidate_id] = resolved.definition
        prior_execution = executions.get(resolved.execution_fingerprint)
        if (
            prior_execution is not None
            and prior_execution != resolved.execution_definition
        ):
            raise ValueError(
                f"execution {resolved.execution_fingerprint} resolved inconsistently"
            )
        executions[resolved.execution_fingerprint] = resolved.execution_definition
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
        required_env.update(candidate_required_env)
        runtime["configuration_sha256"] = stable_digest(runtime)
        prior = runtimes.get(job.candidate_id)
        if prior is not None and prior != runtime:
            raise ValueError(f"candidate {job.candidate_id} runtime binding differs")
        runtimes[job.candidate_id] = runtime

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
    base = RunSnapshotV1(
        schema_version=1,
        identity_schema_version=CANDIDATE_IDENTITY_SCHEMA_VERSION,
        run_id=run_id,
        experiment=_portable(experiment.to_dict(), required_env, secret_names),
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
        },
        required_env=tuple(sorted(name for name in required_env if name)),
        preset=preset,
    )
    serialized = json.dumps(base.to_dict(), sort_keys=True, default=str)
    for name, value in env.items():
        if _SENSITIVE_NAME.search(name) and len(value) >= 8 and value in serialized:
            raise ValueError(f"refusing to serialize runtime secret: {name}")
    digest = stable_digest({**base.to_dict(), "lock_sha256": ""})
    return RunSnapshotV1(**{**asdict(base), "snapshot_sha256": digest})


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
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def verify_snapshot(payload: Mapping[str, Any]) -> bool:
    expected = str(payload.get("snapshot_sha256") or payload.get("lock_sha256") or "")
    unsigned = dict(payload)
    unsigned["snapshot_sha256"] = ""
    unsigned["lock_sha256"] = ""
    return bool(expected) and expected == stable_digest(unsigned)


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
