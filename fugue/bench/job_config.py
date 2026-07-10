from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fugue.bench.library import (
    ExperimentSpec,
    FeatureVariant,
    content_hashes_for_ids,
    get_prompt,
    get_skill,
)
from fugue.bench.manifest import BenchmarkManifest, HarnessSpec
from fugue.bench.memory import write_memory_instruction
from fugue.model_plane import ModelRoute, resolve_model_route, select_model


@dataclass(frozen=True)
class RenderedJob:
    command: list[str]
    config_path: Path
    config: dict[str, Any]
    env: dict[str, str]
    job_name: str
    harness: str
    feature_memory: str
    prompt_id: str | None
    skill_ids: list[str]
    variant_id: str
    variant_label: str
    agent_config_hash: str
    route: ModelRoute


def preview_jobs(
    *,
    experiment: ExperimentSpec,
    manifest: BenchmarkManifest,
    manifest_path: Path,
    repo_root: Path,
    env: dict[str, str],
    model: str | None = None,
    harness_names: list[str] | None = None,
    n_tasks: int | None = None,
    n_attempts: int | None = None,
    n_concurrent: int | None = None,
    jobs_dir: Path | None = None,
    run_name: str | None = None,
    tags: list[str] | None = None,
    run_id: str | None = None,
) -> list[RenderedJob]:
    return _build_jobs(
        experiment=experiment,
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=repo_root,
        env=env,
        model=model,
        harness_names=harness_names,
        n_tasks=n_tasks,
        n_attempts=n_attempts,
        n_concurrent=n_concurrent,
        jobs_dir=jobs_dir,
        run_name=run_name,
        tags=tags,
        run_id=run_id or "preview",
        write_configs=False,
    )


def render_jobs(
    *,
    experiment: ExperimentSpec,
    manifest: BenchmarkManifest,
    manifest_path: Path,
    repo_root: Path,
    env: dict[str, str],
    model: str | None = None,
    harness_names: list[str] | None = None,
    n_tasks: int | None = None,
    n_attempts: int | None = None,
    n_concurrent: int | None = None,
    jobs_dir: Path | None = None,
    run_name: str | None = None,
    tags: list[str] | None = None,
    run_id: str | None = None,
) -> list[RenderedJob]:
    return _build_jobs(
        experiment=experiment,
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=repo_root,
        env=env,
        model=model,
        harness_names=harness_names,
        n_tasks=n_tasks,
        n_attempts=n_attempts,
        n_concurrent=n_concurrent,
        jobs_dir=jobs_dir,
        run_name=run_name,
        tags=tags,
        run_id=run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        write_configs=True,
    )


def _build_jobs(
    *,
    experiment: ExperimentSpec,
    manifest: BenchmarkManifest,
    manifest_path: Path,
    repo_root: Path,
    env: dict[str, str],
    model: str | None,
    harness_names: list[str] | None,
    n_tasks: int | None,
    n_attempts: int | None,
    n_concurrent: int | None,
    jobs_dir: Path | None,
    run_name: str | None,
    tags: list[str] | None,
    run_id: str,
    write_configs: bool,
) -> list[RenderedJob]:
    runtime_dir = repo_root / ".fugue" / "runtime" / run_id / "job-configs"
    if write_configs:
        runtime_dir.mkdir(parents=True, exist_ok=True)

    harnesses = manifest.select_harnesses(harness_names or experiment.harnesses or None)
    variants = [variant for variant in experiment.variants if variant.enabled]
    selected_jobs_dir = jobs_dir or experiment.jobs_dir or manifest.jobs_dir
    selected_attempts = n_attempts or experiment.n_attempts or manifest.k
    selected_concurrent = n_concurrent or experiment.n_concurrent or manifest.n_concurrent
    selected_n_tasks = n_tasks if n_tasks is not None else experiment.n_tasks
    selected_run_name = run_name or experiment.run_name or experiment.id
    selected_tags = [*experiment.tags, *(tags or [])]

    rendered: list[RenderedJob] = []
    for harness in harnesses:
        selected_model = select_model(
            model,
            harness.model or experiment.model or manifest.model,
            env,
        )
        route = resolve_model_route(selected_model, env)
        for variant in variants:
            feature_memory = variant.memory or "none"
            skill_ids = list(variant.skill_ids)
            job_name = _job_name(
                run_name=selected_run_name,
                harness=harness.name,
                variant_id=variant.id,
            )
            agent_config_hash = _agent_config_hash(experiment, variant)
            config = _job_config(
                experiment=experiment,
                variant=variant,
                manifest=manifest,
                harness=harness,
                route=route,
                feature_memory=feature_memory,
                skill_ids=skill_ids,
                agent_config_hash=agent_config_hash,
                job_name=job_name,
                jobs_dir=selected_jobs_dir,
                n_attempts=selected_attempts,
                n_concurrent=selected_concurrent,
                n_tasks=selected_n_tasks,
                repo_root=repo_root,
                write_artifacts=write_configs,
            )
            config_path = runtime_dir / f"{job_name}.json"
            if write_configs:
                config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            job_env = _job_env(
                base_env=env,
                experiment=experiment,
                manifest=manifest,
                manifest_path=manifest_path,
                harness=harness,
                route=route,
                variant=variant,
                feature_memory=feature_memory,
                skill_ids=skill_ids,
                agent_config_hash=agent_config_hash,
                job_name=job_name,
                run_name=selected_run_name,
                tags=selected_tags,
                repo_root=repo_root,
                config_path=config_path,
            )
            rendered.append(
                RenderedJob(
                    command=["harbor", "run", "--config", config_path.as_posix()],
                    config_path=config_path,
                    config=config,
                    env=job_env,
                    job_name=job_name,
                    harness=harness.name,
                    feature_memory=feature_memory,
                    prompt_id=variant.prompt_id,
                    skill_ids=skill_ids,
                    variant_id=variant.id,
                    variant_label=variant.label,
                    agent_config_hash=agent_config_hash,
                    route=route,
                )
            )
    return rendered


def _job_config(
    *,
    experiment: ExperimentSpec,
    variant: FeatureVariant,
    manifest: BenchmarkManifest,
    harness: HarnessSpec,
    route: ModelRoute,
    feature_memory: str,
    skill_ids: list[str],
    agent_config_hash: str,
    job_name: str,
    jobs_dir: Path,
    n_attempts: int,
    n_concurrent: int,
    n_tasks: int | None,
    repo_root: Path,
    write_artifacts: bool,
) -> dict[str, Any]:
    prompt_ids = [variant.prompt_id] if variant.prompt_id else []
    config: dict[str, Any] = {
        "job_name": job_name,
        "jobs_dir": jobs_dir.as_posix(),
        "n_attempts": n_attempts,
        "n_concurrent_trials": n_concurrent,
        "debug": experiment.debug,
        "quiet": experiment.quiet,
        "agents": [
            _agent_config(
                harness=harness,
                route=route,
                experiment=experiment,
                variant=variant,
                skill_ids=skill_ids,
                repo_root=repo_root,
            )
        ],
        "datasets": [
            {
                "name": manifest.dataset.ref,
                "ref": manifest.dataset.version,
                "task_names": [task.id for task in manifest.tasks],
                "n_tasks": n_tasks,
            }
        ],
        "extra_instruction_paths": _extra_instruction_paths(
            repo_root=repo_root,
            artifact_root=manifest.artifact_root,
            feature_memory=feature_memory,
            prompt_ids=prompt_ids,
            write_files=write_artifacts,
        ),
    }
    _set_if(config, "environment", _merge_dicts(experiment.environment, variant.environment))
    _set_if(config, "artifacts", variant.artifacts or experiment.artifacts)
    _set_if(config, "verifier", _merge_dicts(experiment.verifier, variant.verifier))
    _set_if(config, "retry", _merge_dicts(experiment.retry, variant.retry))
    config["fugue"] = {
        "experiment_id": experiment.id,
        "variant_id": variant.id,
        "variant_label": variant.label,
        "prompt_id": variant.prompt_id,
        "feature_memory": feature_memory,
        "skill_ids": skill_ids,
        "agent_config_hash": agent_config_hash,
        "content_hashes": content_hashes_for_ids(
            prompt_ids=prompt_ids,
            skill_ids=skill_ids,
            repo_root=repo_root,
        ),
    }
    return _drop_empty(config)


def _agent_config(
    *,
    harness: HarnessSpec,
    route: ModelRoute,
    experiment: ExperimentSpec,
    variant: FeatureVariant,
    skill_ids: list[str],
    repo_root: Path,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_name": route.display_model,
        "include_logs": ["**/*"],
        "skills": [
            _relative_or_absolute(Path(get_skill(item_id, repo_root).path).parent, repo_root)
            for item_id in skill_ids
        ],
        "kwargs": _merge_dicts(experiment.agent_kwargs, variant.agent_kwargs),
        "env": _merge_dicts(experiment.agent_env, variant.agent_env),
        "mcp_servers": variant.mcp_servers or experiment.mcp_servers,
    }
    if _looks_like_import_path(harness.agent):
        config["import_path"] = harness.agent
        config["name"] = harness.name
    else:
        config["name"] = harness.agent
    return _drop_empty(config)


def _extra_instruction_paths(
    *,
    repo_root: Path,
    artifact_root: Path,
    feature_memory: str,
    prompt_ids: list[str],
    write_files: bool,
) -> list[str]:
    paths: list[Path] = []
    memory_root = _resolve(repo_root, artifact_root)
    memory_instruction = (
        write_memory_instruction(memory_root, feature_memory)
        if write_files
        else _memory_instruction_path(memory_root, feature_memory)
    )
    if memory_instruction is not None:
        paths.append(memory_instruction)
    for item_id in prompt_ids:
        paths.append(Path(get_prompt(item_id, repo_root).path))
    return [_relative_or_absolute(path, repo_root) for path in paths]


def _memory_instruction_path(artifact_root: Path, feature_memory: str) -> Path | None:
    return None if feature_memory == "none" else artifact_root / feature_memory / "INSTRUCTION.md"


def _job_env(
    *,
    base_env: dict[str, str],
    experiment: ExperimentSpec,
    manifest: BenchmarkManifest,
    manifest_path: Path,
    harness: HarnessSpec,
    route: ModelRoute,
    variant: FeatureVariant,
    feature_memory: str,
    skill_ids: list[str],
    agent_config_hash: str,
    job_name: str,
    run_name: str,
    tags: list[str],
    repo_root: Path,
    config_path: Path,
) -> dict[str, str]:
    prompt_ids = [variant.prompt_id] if variant.prompt_id else []
    hashes = content_hashes_for_ids(
        prompt_ids=prompt_ids,
        skill_ids=skill_ids,
        repo_root=repo_root,
    )
    run_tags = _dedupe(
        [
            *_csv(base_env.get("FUGUE_TAGS")),
            "fugue",
            f"experiment-id:{experiment.id}",
            f"variant:{variant.id}",
            f"memory:{feature_memory}",
            *[f"prompt:{item_id}" for item_id in prompt_ids],
            *[f"skill:{item_id}" for item_id in skill_ids],
            f"run:{run_name}",
            f"harness:{harness.name}",
            f"provider:{route.provider}",
            f"model:{route.display_model}",
            *tags,
        ]
    )
    env = dict(base_env)
    env.update(
        {
            "FUGUE_EXPERIMENT_ID": experiment.id,
            "FUGUE_VARIANT_ID": variant.id,
            "FUGUE_FEATURE_MEMORY": feature_memory,
            "FUGUE_PROMPT_ID": ",".join(prompt_ids),
            "FUGUE_PROMPT_HASHES": json.dumps(hashes["prompts"], sort_keys=True),
            "FUGUE_SKILL_IDS": ",".join(skill_ids),
            "FUGUE_SKILL_HASHES": json.dumps(hashes["skills"], sort_keys=True),
            "FUGUE_HARBOR_CONFIG": config_path.as_posix(),
            "FUGUE_AGENT_CONFIG_HASH": agent_config_hash,
            "FUGUE_HARBOR_ENVIRONMENT": str(experiment.environment.get("type") or ""),
            "FUGUE_HARBOR_RESOURCES": json.dumps(
                _resource_summary(experiment.environment), sort_keys=True
            ),
            "FUGUE_RUN_NAME": run_name,
            "FUGUE_RUN_GROUP": env_group(base_env, run_name),
            "FUGUE_TAGS": ",".join(run_tags),
            "FUGUE_MANIFEST_PATH": manifest_path.as_posix(),
            "FUGUE_DATASET": manifest.dataset.harbor_ref,
            "FUGUE_MEMORY_DIR": _resolve(repo_root, manifest.artifact_root).as_posix(),
            "FUGUE_LOCK_PATH": _resolve(repo_root, manifest.lock_path).as_posix(),
            "FUGUE_HARNESS": harness.name,
            "FUGUE_JOB_NAME": job_name,
            "FUGUE_MODEL": route.display_model,
            "FUGUE_MODEL_PROVIDER": route.provider,
            "PYTHONPATH": _prepend_path(repo_root, base_env.get("PYTHONPATH")),
        }
    )
    return env


def env_group(env: dict[str, str], run_name: str) -> str:
    return env.get("FUGUE_RUN_GROUP", "").strip() or run_name


def _agent_config_hash(experiment: ExperimentSpec, variant: FeatureVariant) -> str:
    payload = {
        "agent_kwargs": _merge_dicts(experiment.agent_kwargs, variant.agent_kwargs),
        "agent_env": _merge_dicts(experiment.agent_env, variant.agent_env),
        "mcp_servers": variant.mcp_servers or experiment.mcp_servers,
        "environment": _merge_dicts(experiment.environment, variant.environment),
        "verifier": _merge_dicts(experiment.verifier, variant.verifier),
        "retry": _merge_dicts(experiment.retry, variant.retry),
        "artifacts": variant.artifacts or experiment.artifacts,
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    merged.update(override or {})
    return merged


def _job_name(*, run_name: str, harness: str, variant_id: str) -> str:
    base = f"{_slug(run_name)}-{_slug(harness)}-{_slug(variant_id)}"
    return base[:120].rstrip("-") or "fugue"


def _resource_summary(environment: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "type",
        "cpu_enforcement_policy",
        "memory_enforcement_policy",
        "override_cpus",
        "override_memory_mb",
        "override_storage_mb",
        "override_gpus",
        "override_tpu",
    ]
    return {key: environment[key] for key in keys if key in environment}


def _set_if(config: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, {}, []):
        config[key] = value


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_empty(item)
            for key, item in value.items()
            if item is not None and item != [] and item != {}
        }
    if isinstance(value, list):
        return [_drop_empty(item) for item in value if item is not None]
    return value


def _looks_like_import_path(value: str) -> bool:
    return ":" in value or "." in value


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _prepend_path(path: Path, existing: str | None) -> str:
    root = path.resolve().as_posix()
    return root if not existing else f"{root}:{existing}"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _slug(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    return "-".join(part for part in out.split("-") if part) or "none"
