from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import sys
import webbrowser
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter

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
    mark_unfinished_cells,
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
from fugue.bench.job_config import RenderedJob, preview_jobs, render_jobs
from fugue.bench.library import (
    ExperimentSpec,
    FeatureVariant,
    PresetSpec,
    WorkloadSpec,
    experiment_from_yaml,
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
from fugue.bench.operator import load_env
from fugue.bench.workloads import (
    load_workload_dataset,
    run_retrieval_workload,
    run_sequence_workload,
)
from fugue.bridge import bridge_status, bridge_up, write_bridge_files
from fugue.model_plane import (
    resolve_model_route,
    select_model,
    trace_env_defaults,
    trace_project_slug,
)
from fugue.preflight import PreflightCheck, print_preflight, run_preflight
from fugue.weave_support import trace_async_operation

CONSOLE = Console()


class FugueArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", RichHelpFormatter)
        super().__init__(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = FugueArgumentParser(
        prog="fugue",
        description="Compose and compare Harbor agent experiments in W&B Weave.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run Harbor experiment")
    _add_run_args(run)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--detach", action="store_true")
    run.add_argument("--run-id", help=argparse.SUPPRESS)

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
    preflight.add_argument("--trace-content", choices=("full", "metadata"))
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

    tui = subparsers.add_parser("tui", help="Open the Fugue terminal operator")
    tui.add_argument(
        "--screen",
        choices=("compose", "runs", "results", "setup"),
        default="compose",
    )
    tui.add_argument("--experiment", default="pilot")

    status = subparsers.add_parser("status", help="Show operator readiness")
    status.add_argument("--experiment", default="pilot")
    status.add_argument("--model")
    status.add_argument("--builder-model")
    status.add_argument("--judge-model")
    status.add_argument("--trace-content", choices=("full", "metadata"))
    status.add_argument("--json", action="store_true")
    status.add_argument("--env-file", type=Path, default=Path(".env"))
    status.add_argument("--repo-root", type=Path, default=Path.cwd())

    compose = subparsers.add_parser(
        "compose", help="Draft a Fugue experiment from natural language"
    )
    compose.add_argument("request", nargs="+")
    compose.add_argument("--from", dest="base_experiment", default="pilot")
    compose.add_argument("--model")
    compose.add_argument("--save")
    compose.add_argument("--run", action="store_true")
    compose.add_argument("--replace-assets", action="store_true")
    compose.add_argument("--trace-content", choices=("full", "metadata"))
    compose.add_argument("--json", action="store_true")
    compose.add_argument("--env-file", type=Path, default=Path(".env"))
    compose.add_argument("--repo-root", type=Path, default=Path.cwd())

    analyze = subparsers.add_parser(
        "analyze", help="Analyze Fugue experiments from natural language"
    )
    analyze.add_argument("question", nargs="+")
    analyze.add_argument("--filter", action="append", default=[])
    analyze.add_argument("--model")
    analyze.add_argument("--source", choices=("local", "weave", "hybrid"))
    analyze.add_argument("--save")
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--env-file", type=Path, default=Path(".env"))
    analyze.add_argument("--repo-root", type=Path, default=Path.cwd())

    _add_analysis_commands(subparsers)
    _add_catalog_commands(subparsers)

    _add_runs_commands(subparsers)

    _add_library_commands(subparsers)
    _add_context_commands(subparsers)

    args = parser.parse_args(raw_argv)
    args._raw_argv = raw_argv
    if args.command is None:
        return _tui(argparse.Namespace(screen="compose", experiment="pilot"))
    if args.command == "run":
        if args.detach and not args.dry_run:
            return _detach_run(args)
        return _run(args)
    if args.command == "render":
        return _render(args)
    if args.command == "export":
        return _export(args)
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "bridge":
        return _bridge(args)
    if args.command == "tui":
        return _tui(args)
    if args.command == "status":
        return _status(args)
    if args.command == "compose":
        return _compose_ai(args)
    if args.command == "analyze":
        return _analyze_ai(args)
    if args.command == "analyses":
        return _analyses(args)
    if args.command == "catalog":
        return _catalog(args)
    if args.command == "runs":
        return _runs(args)
    if args.command in {"prompts", "skills", "experiments"}:
        return _library(args)
    if args.command == "context":
        return _context(args)
    raise AssertionError(args.command)


def _add_experiment_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", help="Saved experiment id")
    parser.add_argument(
        "--experiment-file",
        type=Path,
        help="Immutable experiment YAML snapshot",
    )
    parser.add_argument("--manifest", type=Path, help="Benchmark manifest override")


def _add_analysis_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("analyses", help="Manage saved analyses")
    nested = parser.add_subparsers(dest="analyses_command", required=True)
    listing = nested.add_parser("list", help="List saved analyses")
    listing.add_argument("--json", action="store_true")
    show = nested.add_parser("show", help="Show a saved analysis")
    show.add_argument("id")
    run = nested.add_parser("run", help="Run a saved analysis")
    run.add_argument("id")
    run.add_argument("--model")
    run.add_argument("--source", choices=("local", "weave", "hybrid"))
    run.add_argument("--json", action="store_true")
    for item in (listing, show, run):
        item.add_argument("--env-file", type=Path, default=Path(".env"))
        item.add_argument("--repo-root", type=Path, default=Path.cwd())


def _add_catalog_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("catalog", help="Inspect the experiment catalog")
    nested = parser.add_subparsers(dest="catalog_command", required=True)
    refresh = nested.add_parser("refresh", help="Refresh catalog sources")
    refresh.add_argument("--source", choices=("local", "weave", "hybrid"), default="hybrid")
    status = nested.add_parser("status", help="Show catalog status")
    status.add_argument("--json", action="store_true")
    facets = nested.add_parser("facets", help="Show deterministic experiment buckets")
    facets.add_argument("--filter", action="append", default=[])
    facets.add_argument("--json", action="store_true")
    for item in (refresh, status, facets):
        item.add_argument("--env-file", type=Path, default=Path(".env"))
        item.add_argument("--repo-root", type=Path, default=Path.cwd())


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
    parser.add_argument(
        "--trace-content",
        choices=("full", "metadata"),
        help="Weave agent content capture policy (default: experiment or full)",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())


def _add_runs_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("runs", help="Inspect and manage durable runs")
    nested = parser.add_subparsers(dest="runs_command", required=True)
    listing = nested.add_parser("list", help="List recent runs")
    listing.add_argument("--limit", type=_positive_cli_int, default=20)
    listing.add_argument("--json", action="store_true")
    show = nested.add_parser("show", help="Show one run")
    show.add_argument("run_id")
    show.add_argument("--json", action="store_true")
    logs = nested.add_parser("logs", help="Read run or cell logs")
    logs.add_argument("run_id")
    logs.add_argument("--cell")
    logs.add_argument("--follow", action="store_true")
    cancel = nested.add_parser("cancel", help="Cancel a running experiment")
    cancel.add_argument("run_id")
    export = nested.add_parser("export", help="Export one run")
    export.add_argument("run_id")
    export.add_argument("--out", type=Path)
    export.add_argument("--fetch-weave", action="store_true")
    export.add_argument("--to-weave", action="store_true")
    export.add_argument("--republish", action="store_true")
    open_parser = nested.add_parser("open", help="Open W&B or Weave")
    open_parser.add_argument("run_id")
    open_parser.add_argument("--cell")
    open_parser.add_argument(
        "--target", choices=("agents", "trace", "project"), default="agents"
    )
    open_parser.add_argument("--print", action="store_true", dest="print_only")
    for item in (listing, show, logs, cancel, export, open_parser):
        item.add_argument("--repo-root", type=Path, default=Path.cwd())
        item.add_argument("--env-file", type=Path, default=Path(".env"))


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
    run_id = getattr(args, "run_id", None) or new_run_id()
    experiment = _load_experiment_arg(args)
    write_run_manifest(
        args.repo_root,
        run_id,
        {
            "status": "starting",
            "run_name": args.run_name or experiment.run_name or experiment.id,
            "experiment_id": experiment.id,
            "experiment_snapshot": (
                str(args.experiment_file)
                if getattr(args, "experiment_file", None)
                else None
            ),
            "trace_content": args.trace_content or experiment.trace_content,
            "detached": bool(getattr(args, "run_id", None)),
        },
    )
    try:
        if not args.dry_run:
            _materialize_run_datasets(args)
            prepare_args = argparse.Namespace(**vars(args), rebuild=False, run_id=run_id)
            _context_prepare(prepare_args)
        rendered = _rendered_jobs_from_args(args, run_id=run_id)
        for job in rendered:
            if not job.applicable:
                CONSOLE.print(f"[yellow]skip[/] {job.job_name}: {job.skip_reason}")
                continue
            CONSOLE.print("[cyan]+[/] " + " ".join(shlex.quote(part) for part in job.command))
            print(f"# config: {job.config_path}")
        if args.dry_run:
            write_run_manifest(
                args.repo_root,
                run_id,
                {"status": "passed", "dry_run": True, "ended_at": _now()},
            )
            return 0

        run_name = rendered[0].run_name if rendered else _run_name(args.run_name, {})
        cells = plan_cells(rendered, run_id=run_id, run_name=run_name)
        job_dirs = sorted(
            {
                str(job.config.get("jobs_dir"))
                for job in rendered
                if job.config.get("jobs_dir")
            }
        )
        write_run_manifest(
            args.repo_root,
            run_id,
            {
                "status": "running",
                "started_at": _now(),
                "run_name": run_name,
                "experiment_id": experiment.id,
                "routes": _run_route_metadata(rendered),
                "trace_project": (
                    trace_project_slug(rendered[0].env)
                    if rendered
                    else trace_project_slug(load_env(args.env_file))
                ),
                "cell_count": len(cells),
                "jobs_dirs": job_dirs,
                "trace_content": args.trace_content or experiment.trace_content,
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
        status = "failed" if failed else "passed"
        write_run_manifest(
            args.repo_root,
            run_id,
            {
                "status": status,
                "ended_at": _now(),
                "passed_cells": len(outcomes) - failed - skipped,
                "failed_cells": failed,
                "not_applicable_cells": skipped,
            },
        )
        CONSOLE.print(
            f"[bold]run {run_id}[/]: {len(outcomes) - failed - skipped} passed, "
            f"{failed} failed, {skipped} not applicable"
        )
        return 1 if failed else 0
    except KeyboardInterrupt:
        message = "Run interrupted from the terminal."
        mark_unfinished_cells(
            args.repo_root / ".fugue" / "runtime" / run_id,
            "interrupted",
            message=message,
        )
        write_run_manifest(
            args.repo_root,
            run_id,
            {"status": "interrupted", "ended_at": _now(), "error": message},
        )
        return 130
    except Exception as exc:
        write_run_manifest(
            args.repo_root,
            run_id,
            {
                "status": "failed",
                "ended_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


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
    args: argparse.Namespace,
    *,
    run_id: str | None = None,
    write_configs: bool = True,
) -> list[RenderedJob]:
    run_id = run_id or new_run_id()
    experiment = _load_experiment_arg(args)
    experiment = _experiment_with_cli_overrides(experiment, args)
    env = load_env(args.env_file)
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
    run_name = _run_name(args.run_name or experiment.run_name, env)
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
            renderer = render_jobs if write_configs else preview_jobs
            rendered.extend(
                renderer(
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
    inline = getattr(args, "experiment_spec", None)
    if isinstance(inline, ExperimentSpec):
        return inline
    experiment_file = getattr(args, "experiment_file", None)
    if experiment_file:
        path = _resolve(args.repo_root, experiment_file)
        return experiment_from_yaml(path.read_text())
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
        trace_content=getattr(args, "trace_content", None),
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
    env = load_env(args.env_file)
    weave_project = args.weave_project or trace_project_slug(env)
    rows = export_rows(
        args.jobs,
        fetch_weave=args.fetch_weave,
        weave_project=weave_project,
        env=env,
    )
    rows = filter_rows(
        rows,
        presets=_csv(args.preset),
        workloads=_csv(args.workloads),
        systems=_csv(args.systems),
    )
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
            weave_project,
            ledger_root=args.repo_root / ".fugue" / "runtime" / "publications",
            republish=args.republish,
            env=env,
        )
        print(f"published {published} new evaluation row(s) to Weave")
    print(f"exported {len(rows)} rows to {args.out}")
    return 0


def _preflight(args: argparse.Namespace) -> int:
    env = load_env(args.env_file)
    experiment = (
        get_experiment(args.experiment, args.repo_root) if args.experiment else None
    )
    target_model = select_model(
        args.model,
        env=env,
        experiment_model=experiment.model if experiment else None,
    )
    builder_model = (
        args.builder_model
        or (experiment.builder_model if experiment else None)
        or env.get("FUGUE_BUILDER_MODEL")
        or target_model
    )
    judge_model = (
        args.judge_model
        or (experiment.judge_model if experiment else None)
        or env.get("FUGUE_JUDGE_MODEL")
    )
    checks = run_preflight(
        target_model,
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
        trace_content = args.trace_content or experiment.trace_content
        if trace_content == "metadata" and "claude-code" in experiment.harnesses:
            checks.append(
                PreflightCheck(
                    "Claude Code trace content",
                    False,
                    "weave-claude-code cannot guarantee metadata-only capture; "
                    "select full capture or exclude claude-code",
                )
            )
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

    env = load_env(args.env_file)
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


def _detach_run(args: argparse.Namespace) -> int:
    from fugue.bench.supervisor import RunSupervisor

    run_id = getattr(args, "run_id", None) or new_run_id()
    raw = list(args._raw_argv)
    raw = [value for value in raw if value != "--detach"]
    command = [sys.executable, "-m", "fugue.bench.cli", *raw, "--run-id", run_id]
    experiment = _load_experiment_arg(args)
    env = load_env(args.env_file)
    env["PYTHONPATH"] = _prepend_env_path(args.repo_root, env.get("PYTHONPATH"))
    run_name = _run_name(args.run_name or experiment.run_name, env)
    run = RunSupervisor(args.repo_root).start_detached(
        run_id=run_id,
        command=command,
        env=env,
        run_name=run_name,
        experiment_id=experiment.id,
    )
    CONSOLE.print(f"[bold green]started[/] {run.run_id}")
    CONSOLE.print(f"Logs: [cyan]{run.log_path}[/]")
    CONSOLE.print(f"Reattach: [bold]fugue runs logs {run.run_id} --follow[/]")
    return 0


def _tui(args: argparse.Namespace) -> int:
    from fugue.tui import run_tui

    run_tui(initial_screen=args.screen, experiment_id=args.experiment)
    return 0


def _status(args: argparse.Namespace) -> int:
    from fugue.bench.operator import ExperimentRequest, OperatorService, as_json

    service = OperatorService(args.repo_root, args.env_file)
    status = service.status(
        ExperimentRequest(
            experiment_id=args.experiment,
            model=args.model,
            builder_model=args.builder_model,
            judge_model=args.judge_model,
            trace_content=args.trace_content,
        )
    )
    if args.json:
        print(as_json(status))
        return 0
    table = Table(title="Fugue setup", box=None, show_header=False)
    table.add_column("Item", style="bold")
    table.add_column("State")
    table.add_column("Detail", style="dim")
    for route in status.routes:
        _status_row(
            table,
            f"{route.role.title()} model",
            route.key_present,
            f"{route.model} / {route.key_env}",
        )
    _status_row(table, "Weave", status.trace_key_present, status.trace_project)
    _status_row(table, "Docker", status.docker_present, "container runtime")
    _status_row(table, "Harbor", status.harbor_present, "experiment runner")
    _status_row(table, "Bridge", status.bridge_ready, "127.0.0.1:4000")
    _status_row(
        table,
        "Catalog",
        status.catalog_records > 0,
        f"{status.catalog_records} records / {status.catalog_refreshed_at or 'not refreshed'}",
    )
    table.add_row("Trace content", "[yellow]FULL[/]" if status.trace_content == "full" else "metadata", "Prompts and tool data may leave this machine")
    CONSOLE.print(table)
    CONSOLE.print(f"Agents: [link={status.links.agents}]{status.links.agents}[/link]")
    return 0 if all(route.key_present for route in status.routes) and status.trace_key_present else 1


def _compose_ai(args: argparse.Namespace) -> int:
    from fugue.bench.ai import ExperimentComposer
    from fugue.bench.operator import OperatorService, as_json

    service = OperatorService(args.repo_root, args.env_file)
    composer = ExperimentComposer(service)
    draft = asyncio.run(
        composer.compose(
            " ".join(args.request),
            base_experiment=args.base_experiment,
            model=args.model,
            trace_content=args.trace_content,
        )
    )
    saved = None
    run = None
    if args.save:
        saved = composer.save(
            draft,
            experiment_id=args.save,
            replace_assets=args.replace_assets,
        )
    if args.run:
        run = service.launch_experiment(draft.experiment, attached=False)
    if args.json:
        print(
            as_json(
                {
                    "draft": draft,
                    "saved_experiment": saved.id if saved else None,
                    "run": run,
                }
            )
        )
        return 0
    CONSOLE.print(f"[bold]{draft.experiment.title}[/]  [dim]{draft.experiment.id}[/]")
    CONSOLE.print(draft.rationale or "No rationale supplied.")
    CONSOLE.print(
        f"[cyan]{draft.preview.cells} cells[/] / "
        f"{draft.preview.estimated_trials} estimated trials / "
        f"{draft.preview.applicable_cells} applicable"
    )
    if draft.diff:
        CONSOLE.print("\n[bold]Proposed diff[/]")
        CONSOLE.print(draft.diff)
    for warning in draft.warnings:
        CONSOLE.print(f"[yellow]warning[/] {warning}")
    if saved:
        CONSOLE.print(f"[green]saved[/] configs/fugue/experiments/{saved.id}.yaml")
    if run:
        CONSOLE.print(f"[green]started[/] {run.run_id}")
    elif not saved:
        CONSOLE.print("[dim]Draft only. Use --save ID or --run to accept it.[/]")
    return 0


def _analyze_ai(args: argparse.Namespace) -> int:
    from fugue.bench.ai import ExperimentAnalyst, save_analysis
    from fugue.bench.operator import OperatorService, as_json

    service = OperatorService(args.repo_root, args.env_file)
    analyst = ExperimentAnalyst(service)
    result = asyncio.run(
        analyst.analyze(
            " ".join(args.question),
            filters=_key_value_args(args.filter),
            model=args.model,
            source=args.source,
        )
    )
    if args.save:
        save_analysis(
            replace(result.spec, id=args.save, title=result.spec.title),
            args.repo_root,
        )
    if args.json:
        print(as_json(result))
    else:
        CONSOLE.print(result.report)
        CONSOLE.print(f"Report: [cyan]{result.report_dir / 'report.md'}[/]")
    return 0


def _analyses(args: argparse.Namespace) -> int:
    from fugue.bench.ai import ExperimentAnalyst, get_analysis, list_analyses
    from fugue.bench.operator import OperatorService, as_json

    if args.analyses_command == "list":
        values = list_analyses(args.repo_root)
        if args.json:
            print(json.dumps(values, indent=2, sort_keys=True))
        else:
            for item in values:
                print(f"{item['id']}\t{item['title']}\t{item['path']}")
        return 0
    spec = get_analysis(args.id, args.repo_root)
    if args.analyses_command == "show":
        print(yaml.safe_dump(spec.to_dict(), sort_keys=False), end="")
        return 0
    if args.analyses_command == "run":
        if args.source:
            spec = replace(spec, source=args.source)
        service = OperatorService(args.repo_root, args.env_file)
        result = asyncio.run(ExperimentAnalyst(service).analyze(spec=spec, model=args.model))
        if args.json:
            print(as_json(result))
        else:
            CONSOLE.print(result.report)
            CONSOLE.print(f"Report: [cyan]{result.report_dir / 'report.md'}[/]")
        return 0
    raise AssertionError(args.analyses_command)


def _catalog(args: argparse.Namespace) -> int:
    from fugue.bench.catalog import ExperimentCatalog
    from fugue.bench.operator import as_json

    catalog = ExperimentCatalog(args.repo_root, load_env(args.env_file))
    if args.catalog_command == "refresh":
        status = catalog.refresh(source=args.source)
        CONSOLE.print(
            f"[green]catalog refreshed[/] {status.records} records / "
            f"{status.experiments} experiments"
        )
        return 0
    if args.catalog_command == "status":
        status = catalog.status()
        if args.json:
            print(as_json(status))
        else:
            CONSOLE.print(
                f"{status.records} records / {status.experiments} experiments / "
                f"{status.local_records} local / {status.weave_records} Weave"
            )
            CONSOLE.print(f"Catalog: [cyan]{status.path}[/]")
        return 0
    if args.catalog_command == "facets":
        records = catalog.records(filters=_key_value_args(args.filter))
        facets = catalog.facets(records)
        if args.json:
            print(json.dumps(facets, indent=2, sort_keys=True))
        else:
            for name, values in facets.items():
                CONSOLE.print(f"[bold]{name}[/]")
                for value, count in values.items():
                    CONSOLE.print(f"  {value:<30} {count}")
        return 0
    raise AssertionError(args.catalog_command)


def _status_row(table: Table, name: str, ready: bool, detail: str) -> None:
    table.add_row(name, "[green]ready[/]" if ready else "[red]missing[/]", detail)


def _runs(args: argparse.Namespace) -> int:
    from fugue.bench.operator import OperatorService, as_json

    service = OperatorService(args.repo_root, args.env_file)
    if args.runs_command == "list":
        runs = service.runs()[: args.limit]
        if args.json:
            print(as_json(runs))
            return 0
        table = Table(title="Fugue runs", box=None)
        for name in ("Run", "Experiment", "Status", "Passed", "Failed", "Pending"):
            table.add_column(name)
        for run in runs:
            table.add_row(
                run.run_id,
                run.experiment_id,
                _status_markup(run.status),
                str(run.passed),
                str(run.failed),
                str(run.pending),
            )
        if runs:
            CONSOLE.print(table)
        else:
            CONSOLE.print("[dim]No runs yet. Start one with `fugue` or `fugue run`.[/]")
        return 0
    if args.runs_command == "show":
        run = service.run_summary(args.run_id)
        if args.json:
            print(as_json(run))
            return 0
        CONSOLE.print(f"[bold]{run.run_name}[/]  {run.run_id}")
        CONSOLE.print(
            f"{_status_markup(run.status)}  {run.passed} passed  "
            f"{run.failed} failed  {run.pending} pending"
        )
        cells = Table(box=None)
        for name in ("Cell", "Harness", "Variant", "Context", "Task", "Status"):
            cells.add_column(name)
        for cell in run.cells:
            cells.add_row(
                cell.cell_id,
                cell.harness,
                cell.variant_id,
                cell.context_system_id,
                cell.task_id,
                _status_markup(cell.status),
            )
        CONSOLE.print(cells)
        return 0
    if args.runs_command == "logs":
        if args.follow:
            try:
                for chunk in service.supervisor.follow_log(args.run_id, cell_id=args.cell):
                    print(chunk, end="", flush=True)
            except KeyboardInterrupt:
                return 130
        else:
            print(service.supervisor.read_log(args.run_id, cell_id=args.cell), end="")
        return 0
    if args.runs_command == "cancel":
        run = service.supervisor.cancel(args.run_id)
        CONSOLE.print(f"{run.run_id}: {_status_markup(run.status)}")
        return 0
    if args.runs_command == "export":
        run = service.supervisor.get(args.run_id)
        trace_project = str(
            run.metadata.get("trace_project") or trace_project_slug(service.env)
        )
        job_dirs = [
            _resolve(args.repo_root, Path(path))
            for path in run.metadata.get("jobs_dirs", [])
        ]
        sources = [*job_dirs, run.run_dir]
        rows = export_rows(
            sources,
            fetch_weave=args.fetch_weave,
            weave_project=trace_project,
            env=service.env,
        )
        out = args.out or args.repo_root / "reports" / f"{args.run_id}.jsonl"
        write_jsonl(rows, out)
        if args.to_weave:
            published = publish_to_weave(
                rows,
                trace_project,
                ledger_root=args.repo_root / ".fugue" / "runtime" / "publications",
                republish=args.republish,
                env=service.env,
            )
            CONSOLE.print(f"Published {published} new evaluation row(s)")
        CONSOLE.print(f"Exported {len(rows)} rows to [cyan]{out}[/]")
        return 0
    if args.runs_command == "open":
        links = service.run_links(args.run_id)
        url = links.project if args.target == "project" else links.agents
        conversation_id = None
        if args.target == "trace":
            url = links.trace or links.agents
            refs = service.run_trace_refs(args.run_id, cell_id=args.cell)
            conversation_id = next(
                (
                    value
                    for reference in refs
                    for value in reference.conversation_ids
                ),
                None,
            )
        if args.print_only:
            print(url)
        else:
            webbrowser.open(url)
            CONSOLE.print(f"Opened [link={url}]{url}[/link]")
        if conversation_id:
            CONSOLE.print(f"Conversation: [cyan]{conversation_id}[/]")
        return 0
    raise AssertionError(args.runs_command)


def _status_markup(status: str) -> str:
    color = {
        "passed": "green",
        "running": "cyan",
        "starting": "cyan",
        "failed": "red",
        "cancelled": "yellow",
        "interrupted": "yellow",
        "not_applicable": "dim",
    }.get(status, "white")
    return f"[{color}]{status.replace('_', ' ')}[/]"


def _prepend_env_path(root: Path, existing: str | None) -> str:
    value = root.resolve().as_posix()
    return value if not existing else f"{value}{os.pathsep}{existing}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


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
    env = load_env(args.env_file)
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
    env = load_env(args.env_file)
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
        env=load_env(args.env_file),
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


def _key_value_args(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"filter must be FIELD=VALUE: {value}")
        key, selected = value.split("=", 1)
        if not key.strip() or not selected.strip():
            raise ValueError(f"filter must be FIELD=VALUE: {value}")
        result[key.strip()] = selected.strip()
    return result


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


if __name__ == "__main__":
    raise SystemExit(main())
