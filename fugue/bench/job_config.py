from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from fugue.artifacts import artifact_source_paths, harbor_artifacts
from fugue.bench.candidates import (
    CANDIDATE_IDENTITY_SCHEMA_VERSION,
    ResolvedCandidate,
    comparison_example_id,
    resolve_candidate,
)
from fugue.bench.context import (
    DEFAULT_CACHE_ROOT,
    ContextBinding,
    ContextRuntime,
    ContextSystemSpec,
    RepositorySnapshot,
    TrialContext,
    bind_context,
    context_behavior_digest,
    context_cache_key,
    expected_prepared_context,
    get_context_system,
    preflight_context,
    run_async,
)
from fugue.bench.context_contracts import (
    ContextCapability,
    ContextDelivery,
    resolve_context_capabilities,
)
from fugue.bench.evaluations import load_cases, scorer_bundle
from fugue.bench.harness_contracts import harness_capabilities
from fugue.bench.integrations import (
    IntegrationBinding,
    bind_integrations,
    effective_selections,
)
from fugue.bench.library import (
    ExperimentSpec,
    FeatureVariant,
    get_prompt,
)
from fugue.bench.manifest import BenchmarkManifest, HarnessSpec, TaskSpec
from fugue.bench.runtime_manager import read_runtime_lock, render_runtime_compose
from fugue.bench.runtime_provenance import resolve_fugue_source_provenance
from fugue.bench.services import (
    managed_service_environment,
    without_managed_service_environment,
)
from fugue.bench.sources import ResolvedSkill, SkillSetupRequired, resolve_skills
from fugue.model_plane import (
    ModelRoute,
    model_route_identity,
    resolve_model_route,
    select_model,
)
from fugue.preflight import HARBOR_VERSION

CONTEXT_RUNTIME_IMAGE = "fugue-context-runtime:0.1.0"
CONTEXT_RUNTIME_SERVICE = "fugue-context"
CONTEXT_CLIENT_PATH = Path(__file__).resolve().parents[1] / "context_client.py"
PORTABLE_CONTEXT_RUNTIME_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RenderedJob:
    command: list[str]
    config_path: Path
    result_path: Path
    config: dict[str, Any]
    env: dict[str, str]
    job_name: str
    harness: str
    context_system_id: str
    context_delivery: str
    context_version: str
    context_cache_keys: dict[str, str]
    context_cache_ready: bool
    prompt_id: str | None
    skill_ids: list[str]
    variant_id: str
    variant_label: str
    agent_config_hash: str
    route: ModelRoute
    workload_id: str
    preset_id: str | None
    run_id: str
    run_name: str
    task_id: str
    trial_index: int
    n_attempts: int
    comparison_example_id: str
    candidate_id: str
    resolved_candidate: ResolvedCandidate
    execution_kind: str
    evaluation_case: dict[str, Any] | None = None
    evaluation_rubrics: tuple[dict[str, Any], ...] = ()
    scorer_hashes: dict[str, str] | None = None
    scorer_refs: tuple[str, ...] = ()
    applicable: bool = True
    skip_reason: str | None = None
    skill_provenance: tuple[dict[str, Any], ...] = ()
    integration_ids: tuple[str, ...] = ()
    integration_provenance: tuple[dict[str, Any], ...] = ()
    generated_runtime_files: tuple[Path, ...] = ()


def preview_jobs(**kwargs: Any) -> list[RenderedJob]:
    run_id = kwargs.pop("run_id", None) or "preview"
    return _build_jobs(
        **kwargs,
        run_id=run_id,
        write_configs=False,
    )


def render_jobs(**kwargs: Any) -> list[RenderedJob]:
    run_id = kwargs.pop("run_id", None) or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _build_jobs(
        **kwargs,
        run_id=run_id,
        write_configs=True,
    )


def _build_jobs(
    *,
    experiment: ExperimentSpec,
    manifest: BenchmarkManifest,
    manifest_path: Path,
    repo_root: Path,
    env: dict[str, str],
    model: str | None = None,
    harness_names: list[str] | None = None,
    system_names: list[str] | None = None,
    n_tasks: int | None = None,
    n_attempts: int | None = None,
    n_concurrent: int | None = None,
    jobs_dir: Path | None = None,
    run_name: str | None = None,
    tags: list[str] | None = None,
    run_id: str,
    write_configs: bool,
    workload_id: str = "harbor",
    preset_id: str | None = None,
    required_capabilities: list[ContextCapability] | None = None,
    workload_artifacts: list[Any] | None = None,
    scorer_refs: list[str] | None = None,
    asset_overlay: dict[str, str] | None = None,
    source_provenance: dict[str, Any] | None = None,
) -> list[RenderedJob]:
    selected_source_provenance = source_provenance or resolve_fugue_source_provenance(
        repo_root
    )
    runtime_root = repo_root / ".fugue" / "runtime" / run_id
    config_dir = runtime_root / "job-configs"
    if write_configs:
        config_dir.mkdir(parents=True, exist_ok=True)

    harnesses = manifest.select_harnesses(harness_names or experiment.harnesses or None)
    variants = [variant for variant in experiment.variants if variant.enabled]
    if system_names:
        requested = set(system_names)
        variants = [
            variant for variant in variants if variant.context.system_id in requested
        ]
        missing = sorted(
            requested - {variant.context.system_id for variant in variants}
        )
        if missing:
            raise ValueError(
                f"context systems are not variants in this experiment: {', '.join(missing)}"
            )
    selected_jobs_dir = jobs_dir or experiment.jobs_dir or manifest.jobs_dir
    selected_attempts = n_attempts or experiment.n_attempts or manifest.k
    selected_concurrent = (
        n_concurrent or experiment.n_concurrent or manifest.n_concurrent
    )
    selected_n_tasks = n_tasks if n_tasks is not None else experiment.n_tasks
    selected_tasks = (
        manifest.tasks[:selected_n_tasks] if selected_n_tasks else manifest.tasks
    )
    task_groups = [
        ([task], trial_index)
        for task in selected_tasks
        for trial_index in range(1, selected_attempts + 1)
    ]
    selected_run_name = run_name or experiment.run_name or experiment.id
    selected_tags = [*experiment.tags, *(tags or [])]
    runtime = ContextRuntime(
        repo_root=repo_root,
        cache_root=repo_root / DEFAULT_CACHE_ROOT,
        env=env,
    )
    evaluation_cases = _evaluation_cases(
        manifest,
        repo_root,
        asset_overlay or {},
    )
    _, evaluation_rubrics, scorer_hashes = scorer_bundle(
        scorer_refs or [],
        repo_root,
        overlay=asset_overlay,
    )

    rendered: list[RenderedJob] = []
    for harness in harnesses:
        selected_model = select_model(
            model,
            manifest.model,
            env,
            harness_model=harness.model,
            experiment_model=experiment.model,
        )
        route = resolve_model_route(selected_model, env)
        for variant in variants:
            spec = _context_spec(variant, repo_root)
            base_applicable, base_skip_reason = _applicability(
                spec,
                required_capabilities or [],
                runtime,
                variant.context.delivery,
            )
            if experiment.trace_content == "metadata" and harness.name == "claude-code":
                base_applicable = False
                base_skip_reason = (
                    "Claude Code tracing cannot guarantee metadata-only capture; "
                    "use trace_content: full or exclude claude-code"
                )
            skill_ids = variant.skills
            try:
                resolved_skills = resolve_skills(skill_ids, repo_root)
                skill_setup_reason = None
            except SkillSetupRequired as exc:
                resolved_skills = []
                skill_setup_reason = str(exc)
            for tasks, trial_index in task_groups:
                applicable = base_applicable
                skip_reason = base_skip_reason
                job_name = _job_name(
                    run_name=selected_run_name,
                    workload_id=workload_id,
                    harness=harness.name,
                    variant_id=variant.id,
                    task_id=tasks[0].id,
                    trial_index=trial_index,
                )
                binding, cache_keys, cache_ready = _context_binding(
                    spec=spec,
                    variant=variant,
                    tasks=tasks,
                    experiment=experiment,
                    harness=harness,
                    workload_id=workload_id,
                    dataset_id=manifest.dataset.harbor_ref,
                    runtime=runtime,
                    runtime_root=runtime_root,
                    job_name=job_name,
                    write=write_configs,
                )
                integration_binding = bind_integrations(
                    effective_selections(experiment.integrations, variant.integrations),
                    repo_root=repo_root,
                    runtime_root=runtime_root,
                    job_name=job_name,
                    env=env,
                    write=write_configs,
                    reserved_ports=_reserved_context_ports(binding),
                )
                if skill_setup_reason:
                    applicable = False
                    skip_reason = _join_skip_reasons(skip_reason, skill_setup_reason)
                if not integration_binding.applicable:
                    applicable = False
                    skip_reason = _join_skip_reasons(
                        skip_reason, integration_binding.skip_reason
                    )
                selected_mcp = bool(
                    integration_binding.mcp_servers
                    or (
                        binding.mcp_servers and variant.context.delivery == "native_mcp"
                    )
                )
                capabilities = harness_capabilities(harness.agent)
                if applicable and selected_mcp and not capabilities.native_mcp:
                    applicable = False
                    skip_reason = (
                        f"harness adapter {harness.agent} has no reviewed native MCP "
                        "registration contract"
                    )
                agent_config_hash = _agent_config_hash(
                    experiment,
                    variant,
                    spec,
                    binding,
                    resolved_skills,
                    integration_binding,
                )
                context_runtime = _portable_context_runtime_descriptor(
                    binding,
                    variant.context.delivery,
                )
                comparison_example_id = _comparison_example_id(
                    dataset_id=manifest.dataset.harbor_ref,
                    workload_id=workload_id,
                    task_id=tasks[0].id,
                )
                content_hashes = _content_hashes(
                    prompt_ids=[variant.prompt_id] if variant.prompt_id else [],
                    resolved_skills=resolved_skills,
                    repo_root=repo_root,
                )
                resolved_candidate = resolve_candidate(
                    harness=harness.name,
                    model_route=_candidate_model_route(route),
                    prompt_digest=next(iter(content_hashes["prompts"].values()), None),
                    skills=[item.provenance() for item in resolved_skills],
                    context={
                        "id": spec.id,
                        "version": spec.version,
                        "config_hash": _context_config_hash(spec),
                        "delivery": variant.context.delivery,
                    },
                    integrations=integration_binding.identity,
                    agent=_candidate_agent_configuration(experiment, variant),
                    execution={
                        "harbor_version": HARBOR_VERSION,
                        "harbor_config": {
                            "n_attempts": 1,
                            "n_concurrent": selected_concurrent,
                            "verifier": _merge_dicts(
                                experiment.verifier, variant.verifier
                            ),
                            "retry": _merge_dicts(experiment.retry, variant.retry),
                        },
                        "trace_content": experiment.trace_content,
                        "instrumentation": "weave",
                        "fugue_source": selected_source_provenance,
                        **(
                            {"context_runtime": context_runtime}
                            if context_runtime is not None
                            else {}
                        ),
                    },
                )
                candidate_id = resolved_candidate.candidate_id
                context_instruction = _context_instruction_path(
                    runtime_root,
                    spec,
                    delivery=variant.context.delivery,
                    write=write_configs,
                    collect_evidence=bool(required_capabilities),
                )
                config = _job_config(
                    experiment=experiment,
                    variant=variant,
                    manifest=manifest,
                    harness=harness,
                    route=route,
                    context_spec=spec,
                    context_binding=binding,
                    context_instruction=context_instruction,
                    context_cache_keys=cache_keys,
                    skill_ids=skill_ids,
                    resolved_skills=resolved_skills,
                    integration_binding=integration_binding,
                    agent_config_hash=agent_config_hash,
                    job_name=job_name,
                    jobs_dir=selected_jobs_dir,
                    n_attempts=1,
                    n_concurrent=selected_concurrent,
                    tasks=tasks,
                    repo_root=repo_root,
                    workload_id=workload_id,
                    preset_id=preset_id,
                    run_id=run_id,
                    run_name=selected_run_name,
                    trial_index=trial_index,
                    comparison_example_id=comparison_example_id,
                    candidate_id=candidate_id,
                    execution_fingerprint=resolved_candidate.execution_fingerprint,
                    applicable=applicable,
                    skip_reason=skip_reason,
                    collect_evidence=bool(required_capabilities),
                    workload_artifacts=workload_artifacts or [],
                    scorer_hashes=scorer_hashes,
                )
                config_path = config_dir / f"{job_name}.json"
                if write_configs:
                    config_path.write_text(
                        json.dumps(config, indent=2, sort_keys=True) + "\n"
                    )
                job_env = _job_env(
                    base_env=env,
                    experiment=experiment,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    harness=harness,
                    route=route,
                    variant=variant,
                    context_spec=spec,
                    context_binding=binding,
                    context_cache_keys=cache_keys,
                    skill_ids=skill_ids,
                    resolved_skills=resolved_skills,
                    integration_binding=integration_binding,
                    agent_config_hash=agent_config_hash,
                    job_name=job_name,
                    run_name=selected_run_name,
                    tags=selected_tags,
                    repo_root=repo_root,
                    config_path=config_path,
                    workload_id=workload_id,
                    preset_id=preset_id,
                    run_id=run_id,
                    task=tasks[0],
                    task_id=tasks[0].id,
                    trial_index=trial_index,
                    comparison_example_id=comparison_example_id,
                    candidate_id=candidate_id,
                    execution_fingerprint=resolved_candidate.execution_fingerprint,
                    expected_artifact_paths=config.get("fugue", {}).get(
                        "expected_artifact_paths", []
                    ),
                )
                rendered.append(
                    RenderedJob(
                        command=["harbor", "run", "--config", config_path.as_posix()],
                        config_path=config_path,
                        result_path=selected_jobs_dir / job_name / "result.json",
                        config=config,
                        env=job_env,
                        job_name=job_name,
                        harness=harness.name,
                        context_system_id=spec.id,
                        context_delivery=variant.context.delivery,
                        context_version=spec.version,
                        context_cache_keys=cache_keys,
                        context_cache_ready=cache_ready,
                        prompt_id=variant.prompt_id,
                        skill_ids=skill_ids,
                        variant_id=variant.id,
                        variant_label=variant.label,
                        agent_config_hash=agent_config_hash,
                        route=route,
                        workload_id=workload_id,
                        preset_id=preset_id,
                        run_id=run_id,
                        run_name=selected_run_name,
                        task_id=tasks[0].id,
                        trial_index=trial_index,
                        n_attempts=1,
                        comparison_example_id=comparison_example_id,
                        candidate_id=candidate_id,
                        resolved_candidate=resolved_candidate,
                        execution_kind="agent",
                        evaluation_case=evaluation_cases.get(tasks[0].id),
                        evaluation_rubrics=evaluation_rubrics,
                        scorer_hashes=dict(scorer_hashes),
                        scorer_refs=tuple(scorer_refs or []),
                        applicable=applicable,
                        skip_reason=skip_reason,
                        skill_provenance=tuple(
                            item.provenance() for item in resolved_skills
                        ),
                        integration_ids=integration_binding.ids,
                        integration_provenance=integration_binding.provenance,
                        generated_runtime_files=(
                            *binding.compose_files,
                            *integration_binding.compose_files,
                        ),
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
    context_spec: ContextSystemSpec,
    context_binding: ContextBinding,
    context_instruction: Path | None,
    context_cache_keys: dict[str, str],
    skill_ids: list[str],
    resolved_skills: list[ResolvedSkill],
    integration_binding: IntegrationBinding,
    agent_config_hash: str,
    job_name: str,
    jobs_dir: Path,
    n_attempts: int,
    n_concurrent: int,
    tasks: list[TaskSpec],
    repo_root: Path,
    workload_id: str,
    preset_id: str | None,
    run_id: str,
    run_name: str,
    trial_index: int,
    comparison_example_id: str,
    candidate_id: str,
    execution_fingerprint: str,
    applicable: bool,
    skip_reason: str | None,
    collect_evidence: bool,
    workload_artifacts: list[Any],
    scorer_hashes: dict[str, str],
) -> dict[str, Any]:
    prompt_ids = [variant.prompt_id] if variant.prompt_id else []
    environment = _merge_dicts(experiment.environment, variant.environment)
    if context_binding.mounts:
        environment["mounts"] = [
            *environment.get("mounts", []),
            *context_binding.mounts,
        ]
    if context_binding.compose_files:
        environment["extra_docker_compose"] = [
            *environment.get("extra_docker_compose", []),
            *[path.as_posix() for path in context_binding.compose_files],
        ]
    if integration_binding.compose_files:
        environment["extra_docker_compose"] = [
            *environment.get("extra_docker_compose", []),
            *[path.as_posix() for path in integration_binding.compose_files],
        ]
    selected_mcp_servers = [
        *context_binding.mcp_servers,
        *integration_binding.mcp_servers,
    ]
    if _needs_mcp_proxy(selected_mcp_servers):
        mounts = list(environment.get("mounts", []))
        if not any(
            isinstance(item, dict) and item.get("target") == "/fugue-src/fugue"
            for item in mounts
        ):
            mounts.append(_read_only_mount(repo_root / "fugue", "/fugue-src/fugue"))
        environment["mounts"] = mounts
    expected_artifacts = _dedupe_values(
        [
            *(variant.artifacts or experiment.artifacts),
            *(tasks[0].artifacts if tasks else ()),
            *workload_artifacts,
            *context_binding.artifacts,
            *integration_binding.artifacts,
        ]
    )
    artifacts = harbor_artifacts(expected_artifacts)
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
                binding=context_binding,
                resolved_skills=resolved_skills,
                integration_binding=integration_binding,
                repo_root=repo_root,
            )
        ],
        "datasets": [_dataset_config(manifest, repo_root, tasks)],
        "extra_instruction_paths": _extra_instruction_paths(
            repo_root=repo_root,
            context_instruction=context_instruction,
            binding=context_binding,
            integration_binding=integration_binding,
            prompt_ids=prompt_ids,
        ),
    }
    _set_if(config, "environment", environment)
    _set_if(config, "artifacts", artifacts)
    _set_if(config, "verifier", _merge_dicts(experiment.verifier, variant.verifier))
    _set_if(config, "retry", _merge_dicts(experiment.retry, variant.retry))
    config["fugue"] = {
        "experiment_id": experiment.id,
        "run_id": run_id,
        "run_name": run_name,
        "trial_index": trial_index,
        "comparison_example_id": comparison_example_id,
        "candidate_id": candidate_id,
        "execution_fingerprint": execution_fingerprint,
        "workload_id": workload_id,
        "preset_id": preset_id,
        "variant_id": variant.id,
        "variant_label": variant.label,
        "prompt_id": variant.prompt_id,
        "context_system_id": context_spec.id,
        "context_delivery": variant.context.delivery,
        "context_version": context_spec.version,
        "context_support": context_spec.support,
        "context_config_hash": _context_config_hash(context_spec),
        "context_cache_keys": context_cache_keys,
        "skill_ids": skill_ids,
        "skills": [item.provenance() for item in resolved_skills],
        "integration_ids": list(integration_binding.ids),
        "integrations": list(integration_binding.provenance),
        "agent_config_hash": agent_config_hash,
        "applicable": applicable,
        "skip_reason": skip_reason,
        "content_hashes": _content_hashes(
            prompt_ids=prompt_ids,
            resolved_skills=resolved_skills,
            repo_root=repo_root,
        ),
        "expected_evidence_paths": {
            task.id: list(task.expected_paths) for task in tasks if task.expected_paths
        },
        "task_id": tasks[0].id,
        "repository": tasks[0].repo,
        "base_commit": tasks[0].base_commit,
        "repository_source": (
            tasks[0].repository.to_dict() if tasks[0].repository else None
        ),
        "model_provider": route.provider,
        "model": route.display_model,
        "trace_content": experiment.trace_content,
        "harness_capabilities": harness_capabilities(harness.agent).to_dict(),
        "scorer_hashes": scorer_hashes,
        "expected_artifact_paths": artifact_source_paths(expected_artifacts),
    }
    rendered = _drop_empty(config)
    _validate_harbor_job_config(rendered)
    return rendered


def _evaluation_cases(
    manifest: BenchmarkManifest,
    repo_root: Path,
    overlay: dict[str, str],
) -> dict[str, dict[str, Any]]:
    source_path = str(manifest.dataset.source.get("path") or "")
    if not source_path:
        return {}
    relative = Path(source_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evaluation case source must be repository-relative")
    text = overlay.get(relative.as_posix())
    resolved = repo_root / relative
    if text is None and not resolved.is_file():
        return {}
    return {str(case["id"]): case for case in load_cases(resolved, text=text)}


def _dataset_config(
    manifest: BenchmarkManifest,
    repo_root: Path,
    tasks: list[TaskSpec],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "task_names": [_harbor_task_name(manifest, task.id) for task in tasks],
        "n_tasks": len(tasks),
    }
    if manifest.dataset.path:
        path = manifest.dataset.path
        config["path"] = (path if path.is_absolute() else repo_root / path).as_posix()
    else:
        config["name"] = manifest.dataset.ref
        config["ref"] = manifest.dataset.version
    return _drop_empty(config)


def _harbor_task_name(manifest: BenchmarkManifest, task_id: str) -> str:
    """Qualify package task filters while keeping Fugue task ids portable."""
    dataset_ref = manifest.dataset.ref or ""
    if "/" not in dataset_ref or "/" in task_id:
        return task_id
    package_org = dataset_ref.split("/", 1)[0]
    return f"{package_org}/{task_id}"


def _agent_config(
    *,
    harness: HarnessSpec,
    route: ModelRoute,
    experiment: ExperimentSpec,
    variant: FeatureVariant,
    binding: ContextBinding,
    resolved_skills: list[ResolvedSkill],
    integration_binding: IntegrationBinding,
    repo_root: Path,
) -> dict[str, Any]:
    selected_mcp_servers = [
        *binding.mcp_servers,
        *integration_binding.mcp_servers,
    ]
    agent_env = _merge_dicts(
        _merge_dicts(
            _merge_dicts(experiment.agent_env, variant.agent_env), binding.env
        ),
        integration_binding.env,
    )
    if _needs_mcp_proxy(selected_mcp_servers):
        current_pythonpath = str(agent_env.get("PYTHONPATH") or "").strip()
        agent_env["PYTHONPATH"] = (
            f"/fugue-src:{current_pythonpath}" if current_pythonpath else "/fugue-src"
        )
    config: dict[str, Any] = {
        "model_name": route.display_model,
        "include_logs": ["**/*"],
        "skills": [
            _relative_or_absolute(item.path, repo_root) for item in resolved_skills
        ],
        "kwargs": _merge_dicts(experiment.agent_kwargs, variant.agent_kwargs),
        "env": agent_env,
        "mcp_servers": _instrument_mcp_servers(selected_mcp_servers),
        "extra_allowed_hosts": list(integration_binding.allowed_hosts),
    }
    if _looks_like_import_path(harness.agent):
        config["import_path"] = harness.agent
    else:
        config["name"] = harness.agent
    return _drop_empty(config)


def _extra_instruction_paths(
    *,
    repo_root: Path,
    context_instruction: Path | None,
    binding: ContextBinding,
    integration_binding: IntegrationBinding,
    prompt_ids: list[str],
) -> list[str]:
    paths = [*binding.extra_instruction_paths, *integration_binding.instruction_paths]
    if context_instruction is not None:
        paths.insert(0, context_instruction)
    paths.extend(Path(get_prompt(item_id, repo_root).path) for item_id in prompt_ids)
    return [_relative_or_absolute(path, repo_root) for path in paths]


def _job_env(
    *,
    base_env: dict[str, str],
    experiment: ExperimentSpec,
    manifest: BenchmarkManifest,
    manifest_path: Path,
    harness: HarnessSpec,
    route: ModelRoute,
    variant: FeatureVariant,
    context_spec: ContextSystemSpec,
    context_binding: ContextBinding,
    context_cache_keys: dict[str, str],
    skill_ids: list[str],
    resolved_skills: list[ResolvedSkill],
    integration_binding: IntegrationBinding,
    agent_config_hash: str,
    job_name: str,
    run_name: str,
    tags: list[str],
    repo_root: Path,
    config_path: Path,
    workload_id: str,
    preset_id: str | None,
    run_id: str,
    task: TaskSpec,
    task_id: str,
    trial_index: int,
    comparison_example_id: str,
    candidate_id: str,
    execution_fingerprint: str,
    expected_artifact_paths: list[str],
) -> dict[str, str]:
    prompt_ids = [variant.prompt_id] if variant.prompt_id else []
    hashes = _content_hashes(
        prompt_ids=prompt_ids,
        resolved_skills=resolved_skills,
        repo_root=repo_root,
    )
    run_tags = _dedupe(
        [
            *_csv(base_env.get("FUGUE_TAGS")),
            "fugue",
            f"experiment-id:{experiment.id}",
            f"workload:{workload_id}",
            *([f"preset:{preset_id}"] if preset_id else []),
            f"variant:{variant.id}",
            f"context-system:{context_spec.id}",
            *[f"prompt:{item_id}" for item_id in prompt_ids],
            *[f"skill:{item_id}" for item_id in skill_ids],
            *[f"integration:{item_id}" for item_id in integration_binding.ids],
            f"run:{run_name}",
            f"harness:{harness.name}",
            f"provider:{route.provider}",
            f"model:{route.display_model}",
            *tags,
        ]
    )
    env = (
        managed_service_environment(
            base_env,
            repo_root=repo_root,
            target="container",
        )
        if context_spec.id == "graphiti"
        else without_managed_service_environment(base_env)
    )
    env.update(
        {
            "FUGUE_EXPERIMENT_ID": experiment.id,
            "FUGUE_RUN_ID": run_id,
            "FUGUE_WORKLOAD_ID": workload_id,
            "FUGUE_PRESET_ID": preset_id or "",
            "FUGUE_VARIANT_ID": variant.id,
            "FUGUE_CONTEXT_SYSTEM_ID": context_spec.id,
            "FUGUE_CONTEXT_DELIVERY": variant.context.delivery,
            "FUGUE_CONTEXT_VERSION": context_spec.version,
            "FUGUE_CONTEXT_SUPPORT": context_spec.support,
            "FUGUE_CONTEXT_CONFIG_HASH": _context_config_hash(context_spec),
            "FUGUE_CONTEXT_CACHE_KEYS": json.dumps(context_cache_keys, sort_keys=True),
            "FUGUE_CONTEXT_CACHE_ROOT": (repo_root / DEFAULT_CACHE_ROOT).as_posix(),
            "FUGUE_EXPECTED_EVIDENCE_PATHS": json.dumps(
                {
                    task.id: list(task.expected_paths)
                    for task in manifest.tasks
                    if task.expected_paths
                },
                sort_keys=True,
            ),
            "FUGUE_EXPECTED_ARTIFACT_PATHS": json.dumps(
                expected_artifact_paths, sort_keys=True
            ),
            "FUGUE_PROMPT_ID": ",".join(prompt_ids),
            "FUGUE_PROMPT_HASHES": json.dumps(hashes["prompts"], sort_keys=True),
            "FUGUE_SKILL_IDS": ",".join(skill_ids),
            "FUGUE_SKILL_HASHES": json.dumps(hashes["skills"], sort_keys=True),
            "FUGUE_SKILL_PROVENANCE": json.dumps(
                [item.provenance() for item in resolved_skills], sort_keys=True
            ),
            "FUGUE_INTEGRATION_IDS": ",".join(integration_binding.ids),
            "FUGUE_INTEGRATION_PROVENANCE": json.dumps(
                integration_binding.provenance, sort_keys=True
            ),
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
            "FUGUE_HARNESS": harness.name,
            "FUGUE_JOB_NAME": job_name,
            "FUGUE_TASK_NAME": task_id,
            "FUGUE_REPOSITORY": task.repo or "",
            "FUGUE_BASE_COMMIT": task.base_commit or "",
            "FUGUE_TRIAL_INDEX": str(trial_index),
            "FUGUE_COMPARISON_EXAMPLE_ID": comparison_example_id,
            "FUGUE_CANDIDATE_ID": candidate_id,
            "FUGUE_EXECUTION_FINGERPRINT": execution_fingerprint,
            "FUGUE_EXECUTION_KIND": "agent",
            "FUGUE_IDENTITY_SCHEMA_VERSION": str(CANDIDATE_IDENTITY_SCHEMA_VERSION),
            "FUGUE_MODEL": route.display_model,
            "FUGUE_MODEL_PROVIDER": route.provider,
            "FUGUE_TRACE_CONTENT": experiment.trace_content,
            "PYTHONPATH": _prepend_path(repo_root, base_env.get("PYTHONPATH")),
        }
    )
    env.update(
        {
            str(key): str(value)
            for key, value in context_binding.env.items()
            if str(value) != f"${{{key}}}"
        }
    )
    return env


def env_group(env: dict[str, str], run_name: str) -> str:
    return env.get("FUGUE_RUN_GROUP", "").strip() or run_name


def _context_spec(variant: FeatureVariant, repo_root: Path) -> ContextSystemSpec:
    spec = get_context_system(variant.context.system_id, repo_root)
    if not variant.context.config:
        return spec
    return replace(spec, config=_merge_dicts(spec.config, variant.context.config))


def _context_binding(
    *,
    spec: ContextSystemSpec,
    variant: FeatureVariant,
    tasks: list[TaskSpec],
    experiment: ExperimentSpec,
    harness: HarnessSpec,
    workload_id: str,
    dataset_id: str,
    runtime: ContextRuntime,
    runtime_root: Path,
    job_name: str,
    write: bool,
) -> tuple[ContextBinding, dict[str, str], bool]:
    if variant.context.delivery not in spec.deliveries:
        return (
            ContextBinding(),
            {},
            False,
        )
    snapshots = [
        _snapshot_for_task(task, runtime.repo_root, dataset_id) for task in tasks
    ]
    cache_keys = {
        snapshot.task_id: context_cache_key(spec, snapshot, runtime)
        for snapshot in snapshots
    }
    prepared = expected_prepared_context(spec, snapshots[0], runtime)
    trial = TrialContext(
        experiment_id=experiment.id,
        workload_id=workload_id,
        task_id=snapshots[0].task_id,
        harness=harness.name,
    )
    binding = run_async(
        bind_context(
            spec,
            prepared,
            trial,
            runtime,
            delivery=variant.context.delivery,
        )
    )
    if spec.id != "none":
        portable = (
            binding.managed_runtime == "fugue_context"
            and variant.context.delivery == "portable"
        )
        mounts = [] if portable else [_read_only_mount(prepared.path, "/fugue-context")]
        env = dict(binding.env)
        if portable:
            binding = _bind_fugue_context_runtime(
                binding=binding,
                spec=spec,
                prepared=prepared,
                runtime=runtime,
                runtime_root=runtime_root,
                job_name=job_name,
                delivery=variant.context.delivery,
                write=write,
            )
            env = dict(binding.env)
        elif (
            binding.managed_runtime == "pinned_mcp"
            and variant.context.delivery == "native_mcp"
        ):
            if read_runtime_lock(spec.id, runtime.repo_root) is not None:
                compose_path, server, descriptor = render_runtime_compose(
                    spec.id,
                    repo_root=runtime.repo_root,
                    artifact=prepared.path / "artifact",
                    runtime_root=runtime_root,
                    job_name=job_name,
                    env_names=spec.required_env,
                    write=write,
                )
                binding = replace(
                    binding,
                    mcp_servers=(server,),
                    compose_files=(*binding.compose_files, compose_path),
                    runtime_descriptor=descriptor,
                )
            env = dict(binding.env)
        elif binding.mcp_servers:
            mounts.extend(
                [
                    _read_only_mount(
                        runtime.repo_root / "fugue",
                        "/fugue-src/fugue",
                    ),
                    _read_only_mount(
                        runtime.repo_root / "configs" / "fugue" / "context-systems",
                        "/fugue-configs/configs/fugue/context-systems",
                    ),
                ]
            )
            env.update(
                {
                    "FUGUE_REPO_ROOT": "/fugue-configs",
                    "PYTHONPATH": "/fugue-src",
                }
            )
        binding = replace(
            binding,
            mcp_servers=tuple(
                _replace_container_paths(item) for item in binding.mcp_servers
            ),
            env=env,
            mounts=tuple([*binding.mounts, *mounts]),
        )
    cache_ready = all(
        (runtime.cache_root / key / "context-manifest.json").is_file()
        for key in cache_keys.values()
    )
    return binding, cache_keys, cache_ready


def _bind_fugue_context_runtime(
    *,
    binding: ContextBinding,
    spec: ContextSystemSpec,
    prepared: Any,
    runtime: ContextRuntime,
    runtime_root: Path,
    job_name: str,
    delivery: str,
    write: bool,
) -> ContextBinding:
    descriptor = _portable_context_runtime_descriptor(binding, delivery)
    if descriptor is None:
        raise ValueError("managed Fugue context runtime is portable-only here")
    service_name = str(descriptor["service"])
    mcp_port = int(descriptor["mcp_port"])
    portable_port = int(descriptor["portable_port"])
    compose_path = runtime_root / "context-runtime" / f"{job_name}.yaml"
    compose = {
        "services": {
            service_name: {
                "image": descriptor["image"],
                "build": {
                    "context": runtime.repo_root.resolve().as_posix(),
                    "dockerfile": descriptor["dockerfile"],
                },
                "command": [
                    "python",
                    "-m",
                    "fugue.context_server",
                    "--system",
                    spec.id,
                    "--prepared",
                    "/context",
                    "--repo-root",
                    "/opt/fugue",
                    "--transport",
                    "streamable-http",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(mcp_port),
                ],
                "environment": [
                    "ANTHROPIC_API_KEY",
                    "FUGUE_BUILDER_MODEL",
                    "FUGUE_EMBEDDING_MODEL",
                    "FUGUE_GRAPHITI_PASSWORD",
                    "FUGUE_GRAPHITI_URI",
                    "FUGUE_GRAPHITI_USER",
                    "FUGUE_MODEL",
                    "LITELLM_MASTER_KEY",
                    "OPENAI_API_KEY",
                    "WANDB_API_KEY",
                    f"FUGUE_BRIDGE_BASE_URL={descriptor['bridge_url']}",
                    "FUGUE_CONTEXT_EVENTS_PATH=/tmp/fugue-context-events.jsonl",
                ],
                # Portable context is addressed by service name. Keeping it on
                # the project network also leaves the bridge host alias valid.
                "extra_hosts": [descriptor["host_gateway"]],
                "volumes": [
                    {
                        "type": "bind",
                        "source": prepared.path.resolve().as_posix(),
                        "target": "/context",
                        "read_only": True,
                        "bind": {"create_host_path": False},
                    }
                ],
                "healthcheck": {
                    "test": [
                        "CMD",
                        "python",
                        "-c",
                        "import socket; socket.create_connection(("
                        f"'127.0.0.1', {portable_port}"
                        "), 2).close()",
                    ],
                    "interval": "2s",
                    "timeout": "3s",
                    "retries": 30,
                },
            },
        }
    }
    if write:
        compose_path.parent.mkdir(parents=True, exist_ok=True)
        compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))
    return replace(
        binding,
        mcp_servers=(),
        env={
            **binding.env,
            "FUGUE_CONTEXT_COMMAND": "fugue-context",
            "FUGUE_CONTEXT_QUERY_URL": str(descriptor["query_url"]),
            "FUGUE_CONTEXT_EVENTS_PATH": ("/logs/artifacts/fugue-context-events.jsonl"),
        },
        mounts=(
            *binding.mounts,
            _read_only_mount(
                CONTEXT_CLIENT_PATH,
                "/usr/local/bin/fugue-context",
            ),
        ),
        compose_files=(*binding.compose_files, compose_path),
    )


def _portable_context_runtime_descriptor(
    binding: ContextBinding,
    delivery: str,
) -> dict[str, Any] | None:
    if binding.managed_runtime == "pinned_mcp" and delivery == "native_mcp":
        return binding.runtime_descriptor
    if binding.managed_runtime != "fugue_context" or delivery != "portable":
        return None
    return {
        "schema_version": PORTABLE_CONTEXT_RUNTIME_SCHEMA_VERSION,
        "kind": "compose_service",
        "image": CONTEXT_RUNTIME_IMAGE,
        "dockerfile": "Dockerfile.context",
        "service": CONTEXT_RUNTIME_SERVICE,
        "network": "compose_project",
        "host_gateway": "host.docker.internal:host-gateway",
        "bridge_url": "http://host.docker.internal:4000",
        "mcp_port": 8000,
        "portable_port": 8001,
        "query_url": f"http://{CONTEXT_RUNTIME_SERVICE}:8001",
    }


def _reserved_context_ports(binding: ContextBinding) -> dict[int, str]:
    result: dict[int, str] = {}
    for server in binding.mcp_servers:
        parsed = urlparse(str(server.get("url") or ""))
        if parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port:
            result[parsed.port] = str(server.get("name") or "context runtime")
    return result


def _read_only_mount(source: Path, target: str) -> dict[str, Any]:
    return {
        "type": "bind",
        "source": source.resolve().as_posix(),
        "target": target,
        "read_only": True,
        "bind": {"create_host_path": False},
    }


def _replace_container_paths(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(".fugue-context", "/fugue-context")
    if isinstance(value, list):
        return [_replace_container_paths(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _replace_container_paths(item) for key, item in value.items()}
    return value


def _snapshot_for_task(
    task: TaskSpec, repo_root: Path, dataset_id: str
) -> RepositorySnapshot:
    return RepositorySnapshot(
        task_id=task.id,
        repo=task.repo or task.repo_slug,
        commit=task.base_commit or "dataset-managed",
        checkout=repo_root,
        dataset_id=dataset_id,
    )


def _applicability(
    spec: ContextSystemSpec,
    required_capabilities: list[ContextCapability],
    runtime: ContextRuntime,
    delivery: ContextDelivery,
) -> tuple[bool, str | None]:
    resolution = resolve_context_capabilities(
        spec,
        delivery=delivery,
        runner="harbor",
        additional=required_capabilities,
    )
    if not resolution.applicable:
        return False, resolution.reason
    checks = run_async(preflight_context(spec, runtime))
    failed = [
        check
        for check in checks
        if not check.ok
        and check.severity == "required"
        and (check.phase == "runtime" or check.name == "license")
    ]
    if failed:
        return False, "; ".join(f"{item.name}: {item.detail}" for item in failed)
    return True, None


def _context_instruction_path(
    runtime_root: Path,
    spec: ContextSystemSpec,
    *,
    delivery: str,
    write: bool,
    collect_evidence: bool,
) -> Path | None:
    del collect_evidence
    if spec.id == "none":
        return None
    path = runtime_root / "context-instructions" / f"{spec.id}.md"
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        interface = (
            'Query it with `fugue-context query --text "your question" --top-k 10`. '
            if delivery == "portable" and spec.id.startswith("rag-")
            else "Use its injected files or configured native tools when useful. "
        )
        path.write_text(
            "# Fugue Context\n\n"
            f"This trial provides the `{spec.id}` context system through the "
            f"`{delivery}` delivery. {interface}"
            "Verify important evidence against the repository. Fugue records file "
            "and context utilization automatically.\n"
        )
    return path


def _agent_config_hash(
    experiment: ExperimentSpec,
    variant: FeatureVariant,
    spec: ContextSystemSpec,
    binding: ContextBinding,
    resolved_skills: list[ResolvedSkill],
    integration_binding: IntegrationBinding,
) -> str:
    payload = {
        **_candidate_agent_configuration(experiment, variant),
        "derived_mcp_servers": [
            *binding.mcp_servers,
            *integration_binding.mcp_servers,
        ],
        "context": {"id": spec.id, "version": spec.version, "config": spec.config},
        "context_delivery": variant.context.delivery,
        "skills": [item.provenance() for item in resolved_skills],
        "integrations": list(integration_binding.identity),
        "allowed_hosts": list(integration_binding.allowed_hosts),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _candidate_agent_configuration(
    experiment: ExperimentSpec, variant: FeatureVariant
) -> dict[str, Any]:
    return _identity_configuration(
        {
            "agent_kwargs": _merge_dicts(experiment.agent_kwargs, variant.agent_kwargs),
            "agent_env": _merge_dicts(experiment.agent_env, variant.agent_env),
            "environment": _merge_dicts(experiment.environment, variant.environment),
        }
    )


def _identity_configuration(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            tokens = set(name.lower().replace("-", "_").split("_"))
            sensitive = bool(
                tokens & {"token", "secret", "password", "credential"}
                or {"api", "key"} <= tokens
                or {"private", "key"} <= tokens
            )
            result[name] = (
                f"${{{name}}}"
                if sensitive and isinstance(item, str)
                else _identity_configuration(item)
            )
        return result
    if isinstance(value, list | tuple):
        return [_identity_configuration(item) for item in value]
    return value


def _candidate_model_route(route: ModelRoute) -> dict[str, Any]:
    return model_route_identity(route)


def _comparison_example_id(*, dataset_id: str, workload_id: str, task_id: str) -> str:
    return comparison_example_id(
        dataset_id=dataset_id,
        workload_id=workload_id,
        logical_task_id=task_id,
    )


def _stable_id(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _instrument_mcp_servers(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    servers: list[dict[str, Any]] = []
    names: set[str] = set()
    for value in values:
        server = dict(value)
        allowed_tools = [str(item) for item in server.pop("fugue_allowed_tools", [])]
        command = server.get("command")
        name = str(server.get("name") or command or "")
        if name in names:
            raise ValueError(f"duplicate MCP server name: {name}")
        if name:
            names.add(name)
        transport = str(server.get("transport") or ("stdio" if command else "sse"))
        server["transport"] = transport
        if transport != "stdio" or not command:
            servers.append(server)
            continue
        upstream = [str(command), *[str(item) for item in server.get("args", [])]]
        proxy_args = ["--name", str(server.get("name") or command)]
        for tool_name in allowed_tools:
            proxy_args.extend(["--allow-tool", tool_name])
        cwd = server.pop("cwd", None)
        if cwd:
            proxy_args.extend(["--cwd", str(cwd)])
        proxy_args.extend(["--", *upstream])
        server["command"] = "python"
        server["args"] = ["-m", "fugue.mcp_proxy", *proxy_args]
        servers.append(server)
    return servers


def _needs_mcp_proxy(values: list[dict[str, Any]]) -> bool:
    return any(
        str(value.get("transport") or ("stdio" if value.get("command") else "sse"))
        == "stdio"
        and bool(value.get("command"))
        for value in values
    )


def _content_hashes(
    *,
    prompt_ids: list[str],
    resolved_skills: list[ResolvedSkill],
    repo_root: Path,
) -> dict[str, dict[str, str]]:
    return {
        "prompts": {
            item_id: get_prompt(item_id, repo_root).sha256 for item_id in prompt_ids
        },
        "skills": {item.id: item.digest for item in resolved_skills},
    }


def _join_skip_reasons(*values: str | None) -> str | None:
    reasons = [value for value in values if value]
    return "; ".join(dict.fromkeys(reasons)) or None


def _validate_harbor_job_config(config: dict[str, Any]) -> None:
    """Validate against the pinned Harbor schema when installed in this runtime."""
    try:
        actual = version("harbor")
    except PackageNotFoundError:
        return
    if actual != HARBOR_VERSION:
        raise RuntimeError(
            f"Fugue requires harbor=={HARBOR_VERSION}; found harbor=={actual}"
        )
    from harbor.models.job.config import JobConfig

    harbor_config = {key: value for key, value in config.items() if key != "fugue"}
    unknown = sorted(set(harbor_config) - set(JobConfig.model_fields))
    if unknown:
        raise ValueError(
            "generated config contains unknown Harbor field(s): " + ", ".join(unknown)
        )
    JobConfig.model_validate(harbor_config)


def _context_config_hash(spec: ContextSystemSpec) -> str:
    return context_behavior_digest(spec)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    merged.update(override or {})
    return merged


def _job_name(
    *,
    run_name: str,
    workload_id: str,
    harness: str,
    variant_id: str,
    task_id: str | None = None,
    trial_index: int | None = None,
) -> str:
    base = (
        f"{_slug(run_name)}-{_slug(workload_id)}-{_slug(harness)}-{_slug(variant_id)}"
    )
    if task_id:
        base += f"-{_slug(task_id)}"
    suffix = f"-t{trial_index:03d}" if trial_index is not None else ""
    base = base[: 120 - len(suffix)].rstrip("-") or "fugue"
    return f"{base}{suffix}"


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


def _dedupe_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _slug(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    return "-".join(part for part in out.split("-") if part) or "none"
