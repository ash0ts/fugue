from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from fugue.bench.context import (
    DEFAULT_CACHE_ROOT,
    ContextRuntime,
    RepositorySnapshot,
    RetrievalQuery,
    checkout_repository,
    get_context_system,
    list_context_systems,
    preflight_context,
    prepare_context,
    prepared_from_index,
    query_context,
)
from fugue.bench.datasets import materialize_manifest_dataset
from fugue.bench.execution import (
    execute_cells,
    new_run_id,
    plan_cells,
    write_run_manifest,
)
from fugue.bench.export import (
    export_rows,
    filter_rows,
    judge_qa_rows,
    publish_to_weave,
    write_jsonl,
    write_parquet,
)
from fugue.bench.job_config import RenderedJob, render_jobs
from fugue.bench.library import (
    ExperimentSpec,
    FeatureVariant,
    PresetSpec,
    WorkloadSpec,
    experiment_to_yaml,
    experiment_with_overrides,
    get_experiment,
    get_experiment_text,
    get_prompt,
    get_skill,
    list_experiments,
    list_prompts,
    list_skills,
    save_experiment,
    save_prompt,
    save_skill,
)
from fugue.bench.manifest import BenchmarkManifest, load_manifest
from fugue.bench.workloads import (
    load_workload_dataset,
    run_retrieval_workload,
    run_sequence_workload,
)
from fugue.bridge import bridge_status, bridge_up, write_bridge_files
from fugue.model_plane import resolve_model_route, select_model, trace_env_defaults
from fugue.preflight import PreflightCheck, print_preflight, run_preflight
from fugue.weave_support import trace_async_operation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fugue")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run Harbor experiment")
    _add_run_args(run)
    run.add_argument("--dry-run", action="store_true")

    render = subparsers.add_parser("render", help="Render Harbor JobConfig files")
    _add_run_args(render)

    export = subparsers.add_parser("export", help="Export Harbor + Weave rows")
    export.add_argument("--jobs", type=Path, nargs="+", required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--parquet-out", type=Path)
    export.add_argument("--fetch-weave", action="store_true")
    export.add_argument("--to-weave", action="store_true")
    export.add_argument("--republish", action="store_true")
    export.add_argument("--weave-project")
    export.add_argument("--judge-model")
    export.add_argument("--preset")
    export.add_argument("--workloads")
    export.add_argument("--systems")
    export.add_argument("--env-file", type=Path, default=Path(".env"))
    export.add_argument("--repo-root", type=Path, default=Path.cwd())

    preflight = subparsers.add_parser("preflight", help="Validate Fugue setup")
    preflight.add_argument("--model")
    preflight.add_argument("--judge-model")
    preflight.add_argument("--builder-model")
    preflight.add_argument("--experiment")
    preflight.add_argument("--preset")
    preflight.add_argument("--systems")
    preflight.add_argument("--env-file", type=Path, default=Path(".env"))
    preflight.add_argument("--repo-root", type=Path, default=Path.cwd())
    preflight.add_argument("--no-live", action="store_true")
    preflight.add_argument("--no-bridge-up", action="store_true")

    bridge = subparsers.add_parser("bridge", help="Manage the LiteLLM bridge")
    bridge_subparsers = bridge.add_subparsers(dest="bridge_command", required=True)
    bridge_up_parser = bridge_subparsers.add_parser("up", help="Render config and start bridge")
    bridge_up_parser.add_argument("--model")
    bridge_up_parser.add_argument("--builder-model")
    bridge_up_parser.add_argument("--judge-model")
    bridge_up_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    bridge_up_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    bridge_render = bridge_subparsers.add_parser("render", help="Render bridge files only")
    bridge_render.add_argument("--model")
    bridge_render.add_argument("--builder-model")
    bridge_render.add_argument("--judge-model")
    bridge_render.add_argument("--env-file", type=Path, default=Path(".env"))
    bridge_render.add_argument("--repo-root", type=Path, default=Path.cwd())
    bridge_subparsers.add_parser("status", help="Check bridge health")

    web = subparsers.add_parser("web", help="Run local Fugue operator UI")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)

    _add_library_commands(subparsers)
    _add_context_commands(subparsers)

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "render":
        return _render(args)
    if args.command == "export":
        return _export(args)
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "bridge":
        return _bridge(args)
    if args.command == "web":
        return _web(args)
    if args.command in {"prompts", "skills", "experiments"}:
        return _library(args)
    if args.command == "context":
        return _context(args)
    raise AssertionError(args.command)


def _add_experiment_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", help="Saved experiment id")
    parser.add_argument("--manifest", type=Path, help="Benchmark manifest override")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    _add_experiment_arg(parser)
    parser.add_argument("--harnesses", help="Comma-separated harness subset")
    parser.add_argument("--variants", help="Comma-separated variant subset")
    parser.add_argument("--preset", help="Saved experiment preset")
    parser.add_argument("--workloads", help="Comma-separated workload subset")
    parser.add_argument("--systems", help="Comma-separated context-system subset")
    parser.add_argument("--model", help="Model selector: wandb/..., openai/..., anthropic/...")
    parser.add_argument("--judge-model", help="Independent model route for QA judging")
    parser.add_argument("--builder-model", help="Model route used to build generated context")
    parser.add_argument("-k", "--n-attempts", type=_positive_cli_int)
    parser.add_argument("-n", "--n-concurrent", type=_positive_cli_int)
    parser.add_argument("-l", "--n-tasks", type=_positive_cli_int)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--jobs-dir", type=Path)
    parser.add_argument(
        "--run-name",
        help="W&B/Weave run grouping name. Defaults to FUGUE_RUN_NAME or a timestamp.",
    )
    parser.add_argument("--tags", help="Comma-separated extra W&B/Weave tags")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())


def _add_context_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("context", help="Manage context systems")
    nested = parser.add_subparsers(dest="context_command", required=True)
    nested.add_parser("list", help="List configured context systems")
    show = nested.add_parser("show", help="Show one context-system spec")
    show.add_argument("system")
    check = nested.add_parser("preflight", help="Validate context-system dependencies")
    check.add_argument("--systems")
    check.add_argument("--env-file", type=Path, default=Path(".env"))
    check.add_argument("--repo-root", type=Path, default=Path.cwd())
    prepare = nested.add_parser("prepare", help="Build content-addressed context")
    _add_experiment_arg(prepare)
    prepare.add_argument("--systems")
    prepare.add_argument("--workloads")
    prepare.add_argument("--preset")
    prepare.add_argument("--model")
    prepare.add_argument("--builder-model")
    prepare.add_argument("--rebuild", action="store_true")
    prepare.add_argument("--env-file", type=Path, default=Path(".env"))
    prepare.add_argument("--repo-root", type=Path, default=Path.cwd())
    query = nested.add_parser("query", help="Run one directly scored retrieval query")
    query.add_argument("--system", required=True)
    query.add_argument("--task-id", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--expected-paths")
    query.add_argument("--top-k", type=_positive_cli_int, default=10)
    query.add_argument("--repo-root", type=Path, default=Path.cwd())
    evaluate = nested.add_parser("evaluate", help=argparse.SUPPRESS)
    evaluate.add_argument("--experiment", required=True)
    evaluate.add_argument("--workload", required=True)
    evaluate.add_argument("--system", required=True)
    evaluate.add_argument("--preset", required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--attempts", type=_positive_cli_int, default=1)
    evaluate.add_argument("--concurrency", type=_positive_cli_int, default=4)
    evaluate.add_argument("--limit", type=_positive_cli_int)
    evaluate.add_argument("--env-file", type=Path, default=Path(".env"))
    evaluate.add_argument("--repo-root", type=Path, default=Path.cwd())


def _add_library_commands(subparsers: argparse._SubParsersAction) -> None:
    for command, label in (
        ("prompts", "prompt"),
        ("skills", "skill"),
        ("experiments", "experiment"),
    ):
        parser = subparsers.add_parser(command, help=f"Manage saved {label}s")
        nested = parser.add_subparsers(dest="library_command", required=True)
        nested.add_parser("list", help=f"List saved {label}s")
        show = nested.add_parser("show", help=f"Show a saved {label}")
        show.add_argument("id")
        save = nested.add_parser("save", help=f"Save a {label}")
        save.add_argument("id")
        save.add_argument("--file", type=Path)
        save.add_argument("--body")
        validate = nested.add_parser("validate", help=f"Validate a saved {label}")
        validate.add_argument("id")


def _positive_cli_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _run(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    experiment = _load_experiment_arg(args)
    if not args.dry_run:
        _materialize_run_datasets(args)
        prepare_args = argparse.Namespace(**vars(args), rebuild=False, run_id=run_id)
        _context_prepare(prepare_args)
    rendered = _rendered_jobs_from_args(args, run_id=run_id)
    for job in rendered:
        if not job.applicable:
            print(f"# skip {job.job_name}: {job.skip_reason}")
            continue
        print("+ " + " ".join(shlex.quote(part) for part in job.command))
        print(f"# config: {job.config_path}")
    if args.dry_run:
        return 0

    run_name = rendered[0].run_name if rendered else _run_name(args.run_name, {})
    cells = plan_cells(rendered, run_id=run_id, run_name=run_name)
    write_run_manifest(
        args.repo_root,
        run_id,
        {
            "run_name": run_name,
            "experiment_id": experiment.id,
            "routes": _run_route_metadata(rendered),
            "cell_count": len(cells),
        },
    )
    concurrency = args.n_concurrent or experiment.n_concurrent or 2
    outcomes = execute_cells(
        cells,
        repo_root=args.repo_root,
        max_workers=concurrency,
    )
    failed = sum(outcome.status == "failed" for outcome in outcomes)
    skipped = sum(outcome.status == "not_applicable" for outcome in outcomes)
    print(
        f"run {run_id}: {len(outcomes) - failed - skipped} passed, "
        f"{failed} failed, {skipped} not applicable"
    )
    return 1 if failed else 0


def _run_route_metadata(rendered: list[RenderedJob]) -> dict[str, Any]:
    if not rendered:
        return {"target": [], "builder": None, "judge": None}
    first = rendered[0]
    target = sorted(
        {
            (job.route.provider, job.route.display_model, job.route.api_key_env)
            for job in rendered
        }
    )
    return {
        "target": [
            {"provider": provider, "model": model, "api_key_env": key}
            for provider, model, key in target
        ],
        "builder": _resolved_role_metadata(
            first.env.get("FUGUE_BUILDER_MODEL"), first.env
        ),
        "judge": _resolved_role_metadata(
            first.env.get("FUGUE_JUDGE_MODEL"), first.env
        ),
    }


def _resolved_role_metadata(
    model: str | None, env: dict[str, str]
) -> dict[str, str] | None:
    if not model:
        return None
    route = resolve_model_route(model, env)
    return {
        "provider": route.provider,
        "model": route.display_model,
        "api_key_env": route.api_key_env,
    }


def _render(args: argparse.Namespace) -> int:
    rendered = _rendered_jobs_from_args(args, run_id=new_run_id())
    for job in rendered:
        print("+ " + " ".join(shlex.quote(part) for part in job.command))
        print(f"# config: {job.config_path}")
    print(f"rendered {len(rendered)} Harbor job config(s)")
    return 0


def _rendered_jobs_from_args(
    args: argparse.Namespace, *, run_id: str | None = None
) -> list[RenderedJob]:
    run_id = run_id or new_run_id()
    experiment = _load_experiment_arg(args)
    experiment = _experiment_with_cli_overrides(experiment, args)
    env = _load_env(args.env_file)
    env |= trace_env_defaults(env)
    env["FUGUE_BUILDER_MODEL"] = (
        args.builder_model
        or experiment.builder_model
        or env.get("FUGUE_BUILDER_MODEL")
        or args.model
        or experiment.model
        or ""
    )
    if getattr(args, "model", None):
        env["FUGUE_MODEL"] = args.model
    env["FUGUE_JUDGE_MODEL"] = (
        args.judge_model
        or experiment.judge_model
        or env.get("FUGUE_JUDGE_MODEL")
        or ""
    )
    run_name = _run_name(args.run_name, env)
    env["FUGUE_RUN_NAME"] = run_name
    env["FUGUE_RUN_GROUP"] = env.get("FUGUE_RUN_GROUP", "").strip() or run_name
    preset = _selected_preset(experiment, args.preset)
    workloads = _selected_workloads(experiment, preset, _csv(args.workloads))
    if not workloads:
        workloads = [
            WorkloadSpec(
                id="harbor",
                runner="harbor",
                manifest=_manifest_path_from_args(args, experiment),
            )
        ]
    rendered: list[RenderedJob] = []
    for workload in workloads:
        if workload.runner == "harbor":
            manifest_path = _resolve(
                args.repo_root,
                getattr(args, "manifest", None)
                or workload.manifest
                or experiment.manifest,
            )
            manifest = load_manifest(manifest_path)
            env["FUGUE_TAGS"] = ",".join(
                _run_tags(
                    env=env,
                    cli_tags=args.tags,
                    run_name=run_name,
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
            )
            rendered.extend(
                render_jobs(
                    experiment=experiment,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    repo_root=args.repo_root,
                    env=env,
                    model=args.model,
                    harness_names=_csv(args.harnesses) or preset.harnesses,
                    system_names=_selected_system_ids(
                        experiment,
                        workload,
                        preset,
                        _csv(args.systems),
                    ),
                    n_tasks=(
                        args.n_tasks
                        or _preset_workload_int(preset, workload.id, "n_tasks")
                        or workload.n_tasks
                        or preset.n_tasks
                    ),
                    n_attempts=(
                        args.n_attempts
                        or _preset_workload_int(preset, workload.id, "n_attempts")
                        or workload.n_attempts
                        or preset.n_attempts
                    ),
                    n_concurrent=(
                        args.n_concurrent
                        or _preset_workload_int(preset, workload.id, "n_concurrent")
                        or preset.n_concurrent
                    ),
                    jobs_dir=args.jobs_dir,
                    run_name=run_name,
                    tags=_csv(args.tags) or [],
                    run_id=run_id,
                    workload_id=workload.id,
                    preset_id=preset.id if preset.id != "default" else None,
                    required_capabilities=workload.required_capabilities,
                    workload_artifacts=workload.artifacts,
                )
            )
        else:
            rendered.extend(
                _direct_workload_jobs(
                    experiment=experiment,
                    workload=workload,
                    preset=preset,
                    env=env,
                    repo_root=args.repo_root,
                    run_name=run_name,
                    model=args.model,
                    requested_systems=_csv(args.systems),
                    n_tasks=args.n_tasks,
                    n_attempts=args.n_attempts,
                    n_concurrent=args.n_concurrent,
                    run_id=run_id,
                )
            )
    return rendered


def _load_experiment_arg(args: argparse.Namespace) -> ExperimentSpec:
    if getattr(args, "experiment", None):
        return get_experiment(args.experiment, args.repo_root)
    manifest_path = getattr(args, "manifest", None) or Path("datasets/pilot.yaml")
    manifest = load_manifest(manifest_path)
    return ExperimentSpec(
        id=manifest_path.stem,
        title=manifest_path.stem,
        manifest=manifest_path,
        model=manifest.model,
        harnesses=[harness.name for harness in manifest.harnesses],
        variants=[FeatureVariant(id="baseline", label="Baseline")],
        n_attempts=manifest.k,
        n_concurrent=manifest.n_concurrent,
        jobs_dir=manifest.jobs_dir,
    )


def _experiment_with_cli_overrides(
    experiment: ExperimentSpec, args: argparse.Namespace
) -> ExperimentSpec:
    variant_ids = _csv(args.variants) or []
    variants = (
        [variant for variant in experiment.variants if variant.id in set(variant_ids)]
        if variant_ids
        else None
    )
    if variant_ids and not variants:
        raise ValueError(f"unknown variant(s): {', '.join(variant_ids)}")
    return experiment_with_overrides(
        experiment,
        model=args.model,
        builder_model=args.builder_model,
        judge_model=args.judge_model,
        tags=_csv(args.tags),
        harnesses=_csv(args.harnesses),
        variants=[variant.to_dict() for variant in variants] if variants else None,
        n_tasks=args.n_tasks,
        n_attempts=args.n_attempts,
        n_concurrent=args.n_concurrent,
    )


def _manifest_path_from_args(args: argparse.Namespace, experiment: ExperimentSpec) -> Path:
    manifest = getattr(args, "manifest", None)
    return _resolve(args.repo_root, manifest or experiment.manifest)


def _selected_preset(
    experiment: ExperimentSpec, requested: str | None
) -> PresetSpec:
    if not experiment.presets:
        return PresetSpec(id="default")
    preset_id = requested or experiment.default_preset or experiment.presets[0].id
    for preset in experiment.presets:
        if preset.id == preset_id:
            return preset
    raise ValueError(f"unknown preset: {preset_id}")


def _selected_workloads(
    experiment: ExperimentSpec,
    preset: PresetSpec,
    requested: list[str] | None,
) -> list[WorkloadSpec]:
    selected = set(requested or preset.workloads or [item.id for item in experiment.workloads])
    workloads = [item for item in experiment.workloads if item.id in selected]
    missing = sorted(selected - {item.id for item in workloads})
    if missing:
        raise ValueError(f"unknown workload(s): {', '.join(missing)}")
    return workloads


def _preset_workload_int(
    preset: PresetSpec, workload_id: str, key: str
) -> int | None:
    value = (preset.workload_overrides.get(workload_id) or {}).get(key)
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(
            f"preset {preset.id} workload {workload_id} {key} must be positive"
        )
    return parsed


def _selected_system_ids(
    experiment: ExperimentSpec,
    workload: WorkloadSpec,
    preset: PresetSpec,
    requested: list[str] | None,
) -> list[str] | None:
    selected = requested or preset.systems or workload.systems
    if selected:
        allowed = set(workload.systems) if workload.systems else None
        return [item for item in selected if allowed is None or item in allowed]
    values = [
        variant.context.system_id
        for variant in experiment.variants
        if variant.enabled
    ]
    return list(dict.fromkeys(values)) or None


def _direct_workload_jobs(
    *,
    experiment: ExperimentSpec,
    workload: WorkloadSpec,
    preset: PresetSpec,
    env: dict[str, str],
    repo_root: Path,
    run_name: str,
    model: str | None,
    requested_systems: list[str] | None,
    n_tasks: int | None,
    n_attempts: int | None,
    n_concurrent: int | None,
    run_id: str,
) -> list[RenderedJob]:
    if not workload.dataset:
        raise ValueError(f"workload {workload.id} requires dataset")
    dataset_path = _resolve(repo_root, Path(workload.dataset))
    dataset = load_workload_dataset(dataset_path)
    if dataset.runner != workload.runner:
        raise ValueError(
            f"workload {workload.id} runner {workload.runner} does not match {dataset.runner} dataset"
        )
    selected = _selected_system_ids(
        experiment, workload, preset, requested_systems
    ) or []
    selected_model = select_model(model, env=env, experiment_model=experiment.model)
    route = resolve_model_route(selected_model, env)
    direct_env = dict(env)
    direct_env["FUGUE_MODEL"] = selected_model
    attempts = (
        n_attempts
        or _preset_workload_int(preset, workload.id, "n_attempts")
        or workload.n_attempts
        or preset.n_attempts
        or experiment.n_attempts
        or 1
    )
    limit = (
        n_tasks
        or _preset_workload_int(preset, workload.id, "n_tasks")
        or workload.n_tasks
        or preset.n_tasks
    )
    required = set(workload.required_capabilities or [workload.runner])
    runtime = ContextRuntime(
        repo_root=repo_root,
        cache_root=repo_root / DEFAULT_CACHE_ROOT,
        env=env,
    )
    jobs: list[RenderedJob] = []
    for system_id in selected:
        spec = get_context_system(system_id, repo_root)
        missing = sorted(required - set(spec.capabilities))
        license_env = f"FUGUE_LICENSE_APPROVED_{_env_id(system_id)}"
        license_blocked = spec.requires_license_approval and env.get(license_env, "").lower() not in {
            "1",
            "true",
            "yes",
        }
        skip_reason = None
        if missing:
            skip_reason = f"missing context capabilities: {', '.join(missing)}"
        elif license_blocked:
            skip_reason = f"license approval required via {license_env}"
        else:
            failed = [
                check
                for check in asyncio.run(
                    preflight_context(spec, runtime, phase="host")
                )
                if not check.ok and check.severity == "required"
            ]
            if failed:
                skip_reason = "; ".join(
                    f"{check.name}: {check.detail}" for check in failed
                )
        command = [
            sys.executable,
            "-m",
            "fugue.bench.cli",
            "context",
            "evaluate",
            "--experiment",
            experiment.id,
            "--workload",
            workload.id,
            "--system",
            system_id,
            "--preset",
            preset.id,
            "--run-id",
            run_id,
            "--attempts",
            str(attempts),
            "--concurrency",
            str(n_concurrent or preset.n_concurrent or experiment.n_concurrent or 4),
            "--repo-root",
            repo_root.as_posix(),
        ]
        if limit:
            command.extend(["--limit", str(limit)])
        local_count = (
            len(dataset.retrieval_cases)
            if workload.runner == "retrieval"
            else len(dataset.sequence_cases)
        )
        count = int(
            ((dataset.source.get("counts") or {}).get(preset.id)) or local_count
        )
        task_count = min(count, limit) if limit else count
        config = {
            "fugue": {
                "experiment_id": experiment.id,
                "preset_id": preset.id,
                "workload_id": workload.id,
                "runner": workload.runner,
                "context_system_id": system_id,
                "context_version": spec.version,
                "dataset": dataset_path.as_posix(),
                "task_count": task_count,
                "n_attempts": attempts,
                "applicable": skip_reason is None,
                "skip_reason": skip_reason,
            }
        }
        jobs.append(
            RenderedJob(
                command=command,
                config_path=dataset_path,
                config=config,
                env=direct_env,
                job_name=f"{_slug(run_name)}-{workload.id}-{system_id}",
                harness="direct" if workload.runner == "retrieval" else "sequence",
                context_system_id=system_id,
                context_version=spec.version,
                context_cache_keys={},
                context_cache_ready=False,
                prompt_id=None,
                skill_ids=[],
                variant_id=system_id,
                variant_label=spec.title,
                agent_config_hash="",
                route=route,
                workload_id=workload.id,
                preset_id=preset.id,
                run_id=run_id,
                run_name=run_name,
                task_id=dataset.id,
                applicable=skip_reason is None,
                skip_reason=skip_reason,
            )
        )
    return jobs


def _export(args: argparse.Namespace) -> int:
    rows = export_rows(
        args.jobs,
        fetch_weave=args.fetch_weave,
        weave_project=args.weave_project,
    )
    rows = filter_rows(
        rows,
        presets=_csv(args.preset),
        workloads=_csv(args.workloads),
        systems=_csv(args.systems),
    )
    env = _load_env(args.env_file)
    judge_model = args.judge_model or env.get("FUGUE_JUDGE_MODEL")
    if judge_model:
        judge_qa_rows(
            rows,
            model=judge_model,
            env=env,
            repo_root=args.repo_root,
        )
    write_jsonl(rows, args.out)
    parquet_out = args.parquet_out or args.out.with_suffix(".parquet")
    if not write_parquet(rows, parquet_out):
        print("pyarrow is not installed; skipped parquet export")
    if args.to_weave:
        published = publish_to_weave(
            rows,
            args.weave_project,
            ledger_root=args.repo_root / ".fugue" / "runtime" / "publications",
            republish=args.republish,
        )
        print(f"published {published} new evaluation row(s) to Weave")
    print(f"exported {len(rows)} rows to {args.out}")
    return 0


def _preflight(args: argparse.Namespace) -> int:
    env = _load_env(args.env_file)
    experiment = (
        get_experiment(args.experiment, args.repo_root) if args.experiment else None
    )
    builder_model = (
        args.builder_model
        or (experiment.builder_model if experiment else None)
        or env.get("FUGUE_BUILDER_MODEL")
    )
    judge_model = (
        args.judge_model
        or (experiment.judge_model if experiment else None)
        or env.get("FUGUE_JUDGE_MODEL")
    )
    checks = run_preflight(
        args.model,
        repo_root=args.repo_root,
        env=env,
        live=not args.no_live,
        start_bridge=not args.no_bridge_up,
        builder_model=builder_model,
        judge_model=judge_model,
    )
    for role, value in (("builder", builder_model), ("judge", judge_model)):
        if value:
            checks.append(_model_role_check(role, value, env))
    if experiment:
        preset = _selected_preset(experiment, args.preset)
        systems = _csv(args.systems) or preset.systems or [
            variant.context.system_id
            for variant in experiment.variants
            if variant.enabled
        ]
        runtime = ContextRuntime(
            repo_root=args.repo_root,
            cache_root=args.repo_root / DEFAULT_CACHE_ROOT,
            env=env,
        )
        for system_id in dict.fromkeys(systems):
            spec = get_context_system(system_id, args.repo_root)
            for item in asyncio.run(preflight_context(spec, runtime)):
                checks.append(
                    PreflightCheck(
                        f"context {system_id}: {item.name}", item.ok, item.detail
                    )
                )
        if preset.id == "full":
            for workload in _selected_workloads(experiment, preset, None):
                if not workload.dataset:
                    continue
                dataset = load_workload_dataset(
                    _resolve(args.repo_root, Path(workload.dataset))
                )
                command = (dataset.source.get("materialize_command") or [None])[0]
                if command:
                    checks.append(
                        PreflightCheck(
                            f"workload {workload.id} materializer",
                            shutil.which(str(command)) is not None,
                            f"{command} is available"
                            if shutil.which(str(command))
                            else f"install {command} before the full preset",
                        )
                    )
            if not judge_model:
                checks.append(
                    PreflightCheck(
                        "judge model",
                        False,
                        "full preset requires FUGUE_JUDGE_MODEL or --judge-model",
                    )
                )
    return print_preflight(checks)


def _model_role_check(
    role: str,
    model: str,
    env: dict[str, str],
) -> PreflightCheck:
    try:
        route = resolve_model_route(model, env)
    except ValueError as exc:
        return PreflightCheck(f"{role} model", False, str(exc))
    key_present = bool(env.get(route.api_key_env, "").strip())
    return PreflightCheck(
        f"{role} model",
        key_present,
        (
            f"{route.display_model} can use {route.api_key_env}"
            if key_present
            else f"{route.display_model} requires {route.api_key_env}"
        ),
    )


def _bridge(args: argparse.Namespace) -> int:
    if args.bridge_command == "status":
        status = bridge_status()
        print(status)
        return 0 if status.get("ok") else 1

    env = _load_env(args.env_file)
    selected_model = select_model(args.model, env=env)
    route = resolve_model_route(selected_model, env)
    builder_model = args.builder_model or env.get("FUGUE_BUILDER_MODEL")
    judge_model = args.judge_model or env.get("FUGUE_JUDGE_MODEL")
    builder_route = (
        resolve_model_route(builder_model, env) if builder_model else None
    )
    judge_route = resolve_model_route(judge_model, env) if judge_model else None
    if args.bridge_command == "render":
        files = write_bridge_files(
            route,
            args.repo_root,
            builder_route=builder_route,
            judge_route=judge_route,
        )
        print(f"wrote {files.config_path}")
        print(f"wrote {files.compose_path}")
        return 0
    if args.bridge_command == "up":
        files = bridge_up(
            route.display_model,
            repo_root=args.repo_root,
            env=env,
            builder_model=builder_model,
            judge_model=judge_model,
        )
        print(f"bridge running from {files.runtime_dir}")
        return 0
    raise AssertionError(args.bridge_command)


def _web(args: argparse.Namespace) -> int:
    from fugue.web import run_web

    run_web(host=args.host, port=args.port)
    return 0


def _context(args: argparse.Namespace) -> int:
    if args.context_command == "list":
        for spec in list_context_systems():
            default = "default" if spec.enabled_by_default else "opt-in"
            print(
                f"{spec.id}\t{spec.title}\t{spec.version}\t"
                f"{','.join(sorted(spec.capabilities))}\t{default}"
            )
        return 0
    if args.context_command == "show":
        print(yaml.safe_dump(get_context_system(args.system).to_dict(), sort_keys=False), end="")
        return 0
    if args.context_command == "preflight":
        return _context_preflight(args)
    if args.context_command == "prepare":
        return _context_prepare(args)
    if args.context_command == "query":
        return _context_query(args)
    if args.context_command == "evaluate":
        return _context_evaluate(args)
    raise AssertionError(args.context_command)


def _context_preflight(args: argparse.Namespace) -> int:
    env = _load_env(args.env_file)
    runtime = ContextRuntime(
        repo_root=args.repo_root,
        cache_root=args.repo_root / DEFAULT_CACHE_ROOT,
        env=env,
    )
    requested = _csv(args.systems)
    specs = [
        spec
        for spec in list_context_systems(args.repo_root)
        if not requested or spec.id in set(requested)
    ]
    if requested:
        missing = sorted(set(requested) - {spec.id for spec in specs})
        if missing:
            raise ValueError(f"unknown context system(s): {', '.join(missing)}")
    failed = 0
    for spec in specs:
        checks = asyncio.run(preflight_context(spec, runtime))
        status = "ok" if all(check.ok for check in checks) else "FAIL"
        if status == "FAIL":
            failed += 1
        print(f"[{status}] {spec.id} ({spec.version})")
        for check in checks:
            marker = "ok" if check.ok else "FAIL"
            print(f"  [{marker}] {check.name}: {check.detail}")
    return 1 if failed else 0


def _context_prepare(args: argparse.Namespace) -> int:
    env = _load_env(args.env_file)
    experiment = _load_experiment_arg(args)
    env["FUGUE_BUILDER_MODEL"] = (
        getattr(args, "builder_model", None)
        or experiment.builder_model
        or env.get("FUGUE_BUILDER_MODEL")
        or getattr(args, "model", None)
        or experiment.model
        or ""
    )
    runtime = ContextRuntime(
        repo_root=args.repo_root,
        cache_root=args.repo_root / DEFAULT_CACHE_ROOT,
        env=env,
    )
    preset = _selected_preset(experiment, args.preset)
    workloads = _selected_workloads(experiment, preset, _csv(args.workloads))
    for workload in workloads:
        if workload.runner != "harbor":
            continue
        manifest_path = _resolve(
            args.repo_root,
            args.manifest or workload.manifest or experiment.manifest,
        )
        materialized = materialize_manifest_dataset(
            load_manifest(manifest_path),
            args.repo_root,
            rebuild=args.rebuild,
        )
        if materialized:
            print(f"dataset\t{workload.id}\t{materialized}")
    targets = _preparation_targets(
        experiment=experiment,
        workloads=workloads,
        preset=preset,
        requested_systems=_csv(args.systems),
        manifest_override=args.manifest,
        repo_root=args.repo_root,
    )
    records = 0
    checkouts: dict[tuple[str, str, str], RepositorySnapshot] = {}
    for system_id, snapshot in targets:
        spec = get_context_system(system_id, args.repo_root)
        failed = [
            check
            for check in asyncio.run(preflight_context(spec, runtime, phase="host"))
            if not check.ok and check.severity == "required"
        ]
        if failed:
            detail = "; ".join(f"{item.name}: {item.detail}" for item in failed)
            print(f"skip\t{system_id}\t{snapshot.task_id}\t{detail}")
            continue
        checkout = snapshot
        if system_id != "none" and checkout.checkout == args.repo_root:
            snapshot_key = (snapshot.task_id, snapshot.repo, snapshot.commit)
            checkout = checkouts.get(snapshot_key) or checkout_repository(
                task_id=snapshot.task_id,
                repo=snapshot.repo,
                commit=snapshot.commit,
                checkout_root=runtime.cache_root / "checkouts",
                dataset_id=snapshot.dataset_id,
                rebuild=args.rebuild,
            )
            checkouts[snapshot_key] = checkout
        prepared = asyncio.run(
            trace_async_operation(
                "fugue.context.prepare",
                {
                    "experiment_id": experiment.id,
                    "preset_id": preset.id,
                    "run_id": getattr(args, "run_id", None),
                    "context_system_id": system_id,
                    "task_id": snapshot.task_id,
                    "dataset_id": snapshot.dataset_id,
                    "repository": snapshot.repo,
                    "commit": snapshot.commit,
                },
                runtime.env,
                lambda spec=spec, checkout=checkout: prepare_context(
                    spec,
                    checkout,
                    runtime,
                    rebuild=args.rebuild,
                ),
                lambda value: {
                    "cache_key": value.cache_key,
                    "cache_hit": value.cache_hit,
                    **value.metrics,
                },
            )
        )
        status = "cache" if prepared.cache_hit else "built"
        print(
            f"{status}\t{system_id}\t{snapshot.task_id}\t"
            f"{prepared.cache_key[:12]}\t{prepared.path}"
        )
        records += 1
    print(f"prepared {records} context artifact(s)")
    return 0


def _materialize_run_datasets(args: argparse.Namespace) -> None:
    experiment = _load_experiment_arg(args)
    preset = _selected_preset(experiment, args.preset)
    workloads = _selected_workloads(experiment, preset, _csv(args.workloads))
    for workload in workloads:
        if workload.runner != "harbor":
            continue
        manifest_path = _resolve(
            args.repo_root,
            getattr(args, "manifest", None)
            or workload.manifest
            or experiment.manifest,
        )
        path = materialize_manifest_dataset(load_manifest(manifest_path), args.repo_root)
        if path:
            print(f"# dataset ready: {path}")


def _context_query(args: argparse.Namespace) -> int:
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    runtime = ContextRuntime(
        repo_root=args.repo_root,
        cache_root=args.repo_root / DEFAULT_CACHE_ROOT,
        env=dict(os.environ),
    )
    spec = get_context_system(args.system, args.repo_root)
    prepared = prepared_from_index(runtime.cache_root, args.system, args.task_id)
    if prepared is None:
        raise FileNotFoundError(
            f"no prepared context for {args.system}/{args.task_id}; run context prepare"
        )
    hits, metrics = asyncio.run(
        query_context(
            spec,
            RetrievalQuery(
                id="cli",
                text=args.query,
                top_k=args.top_k,
                expected_paths=tuple(_csv(args.expected_paths) or []),
            ),
            prepared,
            runtime,
        )
    )
    print(
        json.dumps(
            {
                "system": spec.id,
                "metrics": metrics,
                "hits": [hit.__dict__ for hit in hits],
            },
            indent=2,
            default=str,
        )
    )
    return 0


def _context_evaluate(args: argparse.Namespace) -> int:
    for name in ("attempts", "concurrency"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    experiment = get_experiment(args.experiment, args.repo_root)
    workload = next(
        (item for item in experiment.workloads if item.id == args.workload), None
    )
    if workload is None or not workload.dataset:
        raise ValueError(f"unknown direct workload: {args.workload}")
    dataset = load_workload_dataset(_resolve(args.repo_root, Path(workload.dataset)))
    runtime = ContextRuntime(
        repo_root=args.repo_root,
        cache_root=args.repo_root / DEFAULT_CACHE_ROOT,
        env=_load_env(args.env_file),
    )
    function = (
        run_retrieval_workload
        if workload.runner == "retrieval"
        else run_sequence_workload
    )
    rows = asyncio.run(
        function(
            dataset=dataset,
            system_id=args.system,
            runtime=runtime,
            experiment_id=experiment.id,
            preset_id=args.preset,
            run_id=args.run_id,
            attempts=args.attempts,
            limit=args.limit,
            **(
                {"concurrency": args.concurrency}
                if workload.runner == "sequence"
                else {}
            ),
        )
    )
    print(f"recorded {len(rows)} {workload.runner} row(s)")
    return 0


def _preparation_targets(
    *,
    experiment: ExperimentSpec,
    workloads: list[WorkloadSpec],
    preset: PresetSpec,
    requested_systems: list[str] | None,
    manifest_override: Path | None,
    repo_root: Path,
) -> list[tuple[str, RepositorySnapshot]]:
    targets: dict[tuple[str, str, str, str], tuple[str, RepositorySnapshot]] = {}
    selected = workloads or [
        WorkloadSpec(
            id="harbor", runner="harbor", manifest=manifest_override or experiment.manifest
        )
    ]
    for workload in selected:
        system_ids = _selected_system_ids(
            experiment,
            workload,
            preset,
            requested_systems,
        ) or []
        required = set(workload.required_capabilities)
        system_ids = [
            system_id
            for system_id in system_ids
            if required
            <= set(get_context_system(system_id, repo_root).capabilities)
        ]
        limit = (
            _preset_workload_int(preset, workload.id, "n_tasks")
            or workload.n_tasks
            or preset.n_tasks
        )
        snapshots: list[RepositorySnapshot] = []
        if workload.runner == "harbor":
            path = _resolve(
                repo_root, manifest_override or workload.manifest or experiment.manifest
            )
            manifest = load_manifest(path)
            tasks = manifest.tasks[:limit] if limit else manifest.tasks
            for task in tasks:
                if not task.repo or not task.base_commit:
                    continue
                snapshots.append(
                    RepositorySnapshot(
                        task.id,
                        task.repo,
                        task.base_commit,
                        repo_root,
                        manifest.dataset.harbor_ref,
                    )
                )
        elif workload.dataset:
            dataset = load_workload_dataset(_resolve(repo_root, Path(workload.dataset)))
            cases = [*dataset.retrieval_cases, *dataset.sequence_cases]
            if limit:
                cases = cases[:limit]
            snapshots.extend(
                RepositorySnapshot(
                    case.id, case.repo, case.commit, repo_root, dataset.id
                )
                for case in cases
            )
        for system_id in system_ids:
            for snapshot in snapshots:
                key = (system_id, snapshot.task_id, snapshot.repo, snapshot.commit)
                targets[key] = (system_id, snapshot)
    return list(targets.values())


def _library(args: argparse.Namespace) -> int:
    if args.command == "prompts":
        return _library_prompt(args)
    if args.command == "skills":
        return _library_skill(args)
    if args.command == "experiments":
        return _library_experiment(args)
    raise AssertionError(args.command)


def _library_prompt(args: argparse.Namespace) -> int:
    if args.library_command == "list":
        _print_items(list_prompts())
        return 0
    if args.library_command == "show":
        print(get_prompt(args.id).body, end="")
        return 0
    if args.library_command == "save":
        item = save_prompt(args.id, _body_arg(args))
        print(f"saved prompt {item.id} -> {item.path}")
        return 0
    if args.library_command == "validate":
        item = get_prompt(args.id)
        print(json.dumps({"id": item.id, "sha256": item.sha256}, sort_keys=True))
        return 0
    raise AssertionError(args.library_command)


def _library_skill(args: argparse.Namespace) -> int:
    if args.library_command == "list":
        _print_items(list_skills())
        return 0
    if args.library_command == "show":
        print(get_skill(args.id).body, end="")
        return 0
    if args.library_command == "save":
        item = save_skill(args.id, _body_arg(args))
        print(f"saved skill {item.id} -> {item.path}")
        return 0
    if args.library_command == "validate":
        item = get_skill(args.id)
        print(json.dumps({"id": item.id, "sha256": item.sha256}, sort_keys=True))
        return 0
    raise AssertionError(args.library_command)


def _library_experiment(args: argparse.Namespace) -> int:
    if args.library_command == "list":
        _print_items(list_experiments())
        return 0
    if args.library_command == "show":
        print(get_experiment_text(args.id), end="")
        return 0
    if args.library_command == "save":
        experiment = save_experiment(args.id, _body_arg(args))
        print(f"saved experiment {experiment.id}")
        return 0
    if args.library_command == "validate":
        experiment = get_experiment(args.id)
        print(experiment_to_yaml(experiment), end="")
        return 0
    raise AssertionError(args.library_command)


def _print_items(items) -> None:
    for item in items:
        print(f"{item.id}\t{item.title}\t{item.sha256[:12]}\t{item.path}")


def _body_arg(args: argparse.Namespace) -> str:
    if args.file:
        return args.file.read_text()
    if args.body is not None:
        return args.body
    raise ValueError("--file or --body is required")


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _run_name(cli_value: str | None, env: dict[str, str]) -> str:
    value = cli_value or env.get("FUGUE_RUN_NAME")
    if value and value.strip():
        return value.strip()
    return datetime.now(UTC).strftime("fugue-%Y%m%dT%H%M%SZ")


def _run_tags(
    *,
    env: dict[str, str],
    cli_tags: str | None,
    run_name: str,
    manifest: BenchmarkManifest,
    manifest_path: Path,
) -> list[str]:
    configured = []
    for raw in (env.get("FUGUE_TAGS"), cli_tags):
        configured.extend(_csv(raw) or [])
    return _dedupe(
        [
            "fugue",
            f"run:{run_name}",
            f"dataset:{manifest.dataset.ref or manifest.dataset.path}",
            f"manifest:{manifest_path.stem}",
            *configured,
        ]
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _slug(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    return "-".join(part for part in out.split("-") if part) or "fugue"


def _env_id(value: str) -> str:
    return "".join(ch.upper() if ch.isalnum() else "_" for ch in value)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_env(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key not in env:
            env[key] = value.strip().strip("'\"")
    return env


if __name__ == "__main__":
    raise SystemExit(main())
