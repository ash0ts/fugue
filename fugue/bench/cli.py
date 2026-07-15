from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
import time
import webbrowser
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich_argparse import RichHelpFormatter

from fugue.bench.context import (
    DEFAULT_CACHE_ROOT,
    ContextRuntime,
)
from fugue.bench.execution import (
    execute_cells,
    mark_unfinished_cells,
    new_run_id,
    plan_cells,
    write_run_manifest,
)
from fugue.bench.export import LiveEvaluationCoordinator
from fugue.bench.job_config import RenderedJob
from fugue.bench.library import (
    ExperimentSpec,
    FeatureVariant,
    experiment_from_yaml,
    get_experiment,
)
from fugue.bench.manifest import load_manifest
from fugue.bench.operator import (
    ExperimentRequest,
    OperatorService,
    load_env,
)
from fugue.bench.workloads import (
    load_workload_dataset,
    run_retrieval_workload,
    run_sequence_workload,
)
from fugue.model_plane import (
    resolve_model_route,
    trace_project_slug,
)

FUGUE_THEME = Theme(
    {
        "fugue.gold": "#FFCC33",
        "fugue.cyan": "#00AFC2",
        "fugue.coral": "#FF6B6B",
        "fugue.success": "#22C55E",
        "fugue.muted": "#9CA3AF",
    }
)
CONSOLE = Console(theme=FUGUE_THEME)


class FugueArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", RichHelpFormatter)
        super().__init__(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ["_context-evaluate"]:
        return _internal_context_evaluate(raw_argv[1:])
    parser = _parser()
    args = parser.parse_args(raw_argv)
    args._raw_argv = raw_argv
    if args.command is None:
        return _command_center(parser)
    return int(args.handler(args))


def _internal_context_evaluate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempts", type=_positive_cli_int, default=1)
    parser.add_argument("--concurrency", type=_positive_cli_int, default=4)
    parser.add_argument("--limit", type=_positive_cli_int)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return _context_evaluate(parser.parse_args(argv))


def _parser() -> FugueArgumentParser:
    parser = FugueArgumentParser(
        prog="fugue",
        description="Plan, run, and analyze Harbor agent experiments in W&B Weave.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    plan = subparsers.add_parser("plan", help="Plan an experiment with Fugue AI")
    plan.add_argument("request", nargs="+")
    plan.add_argument("--from", dest="base_experiment", default="pilot")
    plan.add_argument("--model")
    plan.add_argument("--save")
    plan.add_argument("--run", action="store_true")
    plan.add_argument("--yes", action="store_true")
    plan.add_argument("--replace-assets", action="store_true")
    plan.add_argument("--trace-content", choices=("full", "metadata"))
    _add_common_args(plan, json_output=True)
    plan.set_defaults(handler=_plan)

    run = subparsers.add_parser("run", help="Preview or run an experiment")
    run.add_argument("experiment", nargs="?", help="Saved experiment id (default: pilot)")
    _add_run_args(run)
    run.add_argument("--preview", action="store_true", help="Show the matrix without writing runtime state")
    run.add_argument("--detach", action="store_true", help="Start the durable run and return immediately")
    run.add_argument("--json", action="store_true", help="Emit structured output without Rich decoration")
    run.add_argument("--run-id", help=argparse.SUPPRESS)
    run.add_argument("--experiment-file", type=Path, help=argparse.SUPPRESS)
    run.set_defaults(handler=_run_command)

    runs = subparsers.add_parser("runs", help="Inspect and manage durable runs")
    runs.add_argument("run_id", nargs="?", help="Managed run id; omit to list recent runs")
    runs.add_argument("--limit", type=_positive_cli_int, default=20, help="Maximum recent runs to list")
    action = runs.add_mutually_exclusive_group()
    action.add_argument("--logs", action="store_true", help="Read run or selected-cell logs")
    action.add_argument("--cancel", action="store_true", help="Cancel the managed process group")
    action.add_argument("--export", action="store_true", dest="export_run", help="Write normalized JSONL")
    action.add_argument("--open", choices=("agents", "trace", "project"), help="Open the stable W&B destination")
    runs.add_argument("--follow", action="store_true", help="Follow logs until the run finishes")
    runs.add_argument("--cell", help="Select one cell for logs or trace lookup")
    runs.add_argument("--out", type=Path, help="Export destination")
    runs.add_argument("--fetch-weave", action="store_true", help="Enrich exported rows with Weave summaries")
    runs.add_argument("--to-weave", action="store_true", help="Publish normalized evaluations once")
    runs.add_argument("--republish", action="store_true", help="Intentionally bypass the publication ledger")
    runs.add_argument("--print", action="store_true", dest="print_only", help="Print a link instead of opening it")
    _add_common_args(runs, json_output=True)
    runs.set_defaults(handler=_runs)

    analyze = subparsers.add_parser("analyze", help="Analyze experiment results with Fugue AI")
    analyze.add_argument("question", nargs="*")
    source = analyze.add_mutually_exclusive_group()
    source.add_argument("--saved", help="Run a saved analysis definition")
    source.add_argument("--list", action="store_true", dest="list_saved", help="List saved analysis definitions")
    analyze.add_argument("--filter", action="append", default=[], help="Required FIELD=VALUE scope filter")
    analyze.add_argument("--model", help="Analyst model route")
    analyze.add_argument("--source", choices=("local", "hybrid"), help="Use local outcomes or narrow Weave enrichment")
    analyze.add_argument("--save", help="Save the generated analysis definition")
    analyze.add_argument("--yes", action="store_true", help="Confirm report generation without prompting")
    _add_common_args(analyze, json_output=True)
    analyze.set_defaults(handler=_analyze)

    setup = subparsers.add_parser("setup", help="Inspect and prepare Fugue dependencies")
    setup.add_argument("--experiment", default="pilot")
    setup.add_argument("--model")
    setup.add_argument("--builder-model")
    setup.add_argument("--judge-model")
    setup.add_argument("--preset")
    setup.add_argument("--manifest", type=Path)
    setup.add_argument("--workloads")
    setup.add_argument("--systems")
    setup.add_argument("--trace-content", choices=("full", "metadata"))
    operation = setup.add_mutually_exclusive_group()
    operation.add_argument("--check", action="store_true", help="Run observational live preflight")
    operation.add_argument("--start-bridge", action="store_true", help="Start the local LiteLLM bridge")
    operation.add_argument("--prepare-context", action="store_true", help="Build selected context artifacts")
    setup.add_argument("--rebuild", action="store_true", help="Ignore reusable context cache entries")
    _add_common_args(setup, json_output=True)
    setup.set_defaults(handler=_setup)

    tui = subparsers.add_parser("tui", help="Open the full-screen terminal workspace")
    tui.add_argument("--screen", choices=("plan", "runs", "results", "setup"), default="plan")
    tui.add_argument("--experiment", default="pilot")
    tui.set_defaults(handler=_tui)
    return parser


def _add_common_args(parser: argparse.ArgumentParser, *, json_output: bool = False) -> None:
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    if json_output:
        parser.add_argument("--json", action="store_true", help="Emit structured output without Rich decoration")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, help="Benchmark manifest override")
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
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help=argparse.SUPPRESS)
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
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)


def _positive_cli_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _command_center(parser: FugueArgumentParser) -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        parser.print_help()
        return 0
    service = OperatorService()
    while True:
        CONSOLE.clear()
        _print_home(service)
        action = Prompt.ask(
            "Action",
            choices=("plan", "analyze", "run", "tui", "setup", "exit"),
            default="plan",
        )
        if action == "exit":
            return 0
        if action == "tui":
            from fugue.tui import run_tui

            run_tui(service=service)
            continue
        if action == "setup":
            _print_setup(service.status())
        elif action == "plan":
            request = Prompt.ask("Describe the experiment")
            base = Prompt.ask("Base experiment", default="pilot")
            main(["plan", request, "--from", base])
        elif action == "analyze":
            question = Prompt.ask("What should Fugue analyze?")
            main(["analyze", question])
        elif action == "run":
            experiment = Prompt.ask("Experiment", default="pilot")
            main(["run", experiment])
        Prompt.ask("Press enter to return", default="")


def _print_home(service: OperatorService) -> None:
    status = service.status()
    runs = service.runs()
    latest = runs[0] if runs else None
    title = Text("FUGUE", style="bold fugue.gold")
    title.append("  AGENT EXPERIMENT OPERATOR", style="fugue.muted")
    identity = Table.grid(padding=(0, 2))
    identity.add_column(style="bold")
    identity.add_column()
    identity.add_row("Model", f"{status.model}  [{status.model_provider}]")
    identity.add_row("Weave", status.trace_project)
    identity.add_row("Experiments", str(status.experiments))
    ready = sum(
        (
            status.model_key_present,
            status.trace_key_present,
            status.docker_present,
            status.harbor_present,
        )
    )
    readiness = Table.grid(padding=(0, 2))
    readiness.add_row("Credentials", _state(status.model_key_present and status.trace_key_present))
    readiness.add_row("Docker", _state(status.docker_present))
    readiness.add_row("Harbor", _state(status.harbor_present))
    readiness.add_row("Bridge", _state(status.bridge_ready))
    latest_text = (
        f"[bold]{latest.run_name}[/]\n{_status_markup(latest.status)}\n"
        f"{latest.passed} passed  {latest.failed} failed"
        if latest
        else "[fugue.muted]No runs yet[/]"
    )
    CONSOLE.print(Panel(title, border_style="fugue.gold", box=box.SQUARE))
    CONSOLE.print(
        Columns(
            (
                Panel(identity, title="Workspace", border_style="fugue.cyan"),
                Panel(readiness, title=f"Readiness {ready}/4", border_style="fugue.gold"),
                Panel(latest_text, title="Latest run", border_style="fugue.coral"),
            ),
            equal=True,
            expand=True,
        )
    )
    if latest:
        CONSOLE.print(_sequencer(latest))
    CONSOLE.print(
        "[bold fugue.gold]plan[/] experiment   "
        "[bold fugue.cyan]analyze[/] results   "
        "[bold]run[/] experiment   [bold]tui[/] workspace   [bold]setup[/]"
    )


def _state(ready: bool) -> str:
    return "[fugue.success]ready[/]" if ready else "[fugue.coral]missing[/]"


def _sequencer(run: Any) -> Panel:
    statuses = {cell.harness: cell.status for cell in run.cells}
    lines = []
    for label, harness in (
        ("HERMES", "hermes"),
        ("OPENCLAW", "openclaw"),
        ("CLAUDE", "claude-code"),
        ("CODEX", "codex"),
    ):
        status = statuses.get(harness, "pending")
        glyph = "■" if status == "running" else "▪" if status == "passed" else "×" if status == "failed" else "·"
        lines.append(f"{label:<10} {glyph * 12}  {status.replace('_', ' ')}")
    return Panel("\n".join(lines), title="Harness sequencer", border_style="fugue.cyan")


def _print_preview(preview: Any) -> None:
    summary = Table.grid(padding=(0, 2))
    summary.add_row("Cells", str(preview.cells))
    summary.add_row("Applicable", str(preview.applicable_cells))
    summary.add_row("Estimated trials", str(preview.estimated_trials))
    summary.add_row("Harnesses", ", ".join(preview.harnesses) or "none")
    summary.add_row("Variants", ", ".join(preview.variants) or "none")
    summary.add_row("Workloads", ", ".join(preview.workloads) or "none")
    commands = "\n".join(preview.commands) or "No applicable commands."
    CONSOLE.print(
        Group(
            Panel(summary, title="Experiment matrix", border_style="fugue.gold"),
            Panel(Syntax(commands, "bash", word_wrap=True), title="Harbor commands", border_style="fugue.cyan"),
        )
    )


def _print_draft(draft: Any) -> None:
    body = [draft.rationale or "No rationale supplied."]
    if draft.assumptions:
        body.append("\nAssumptions: " + "; ".join(draft.assumptions))
    if draft.assets:
        body.append("\nAssets: " + ", ".join(f"{item.kind}:{item.id}" for item in draft.assets))
    CONSOLE.print(
        Panel(
            "\n".join(body),
            title=f"{draft.experiment.title}  [{draft.experiment.id}]",
            border_style="fugue.gold",
        )
    )
    _print_preview(draft.preview)
    if draft.diff:
        CONSOLE.print(Panel(Syntax(draft.diff, "diff"), title="Proposed diff", border_style="fugue.cyan"))
    for warning in draft.warnings:
        CONSOLE.print(f"[fugue.coral]warning[/] {warning}")


def _print_analysis_preview(preview: Any) -> None:
    scope = preview.scope
    table = Table.grid(padding=(0, 2))
    table.add_row("Experiments", ", ".join(scope.experiments) or "none")
    table.add_row("Runs", str(len(scope.runs)))
    table.add_row("Trial records", str(scope.rows))
    table.add_row("Tasks", str(len(scope.tasks)))
    table.add_row("Models", ", ".join(scope.models) or "none")
    table.add_row("Variants", ", ".join(scope.variants) or "none")
    table.add_row("Sources", ", ".join(scope.sources) or "local")
    if scope.missing_metrics:
        table.add_row("Missing metrics", ", ".join(scope.missing_metrics))
    CONSOLE.print(Panel(table, title="Resolved analysis scope", border_style="fugue.cyan"))
    for warning in scope.warnings:
        CONSOLE.print(f"[fugue.coral]warning[/] {warning}")


def _print_setup(status: Any) -> None:
    table = Table("Component", "State", "Detail", box=box.SIMPLE_HEAD)
    for route in status.routes:
        table.add_row(
            f"{route.role.title()} model",
            _state(route.key_present),
            f"{route.model} / {route.key_env}",
        )
    table.add_row("Weave", _state(status.trace_key_present), status.trace_project)
    table.add_row("Docker", _state(status.docker_present), "container runtime")
    table.add_row("Harbor", _state(status.harbor_present), "experiment runner")
    table.add_row("Bridge", _state(status.bridge_ready), "127.0.0.1:4000")
    table.add_row(
        "Context cache",
        str(status.context_cache_entries),
        ", ".join(status.selected_context_systems) or "no selected systems",
    )
    table.add_row(
        "Trace content",
        "[fugue.coral]FULL[/]" if status.trace_content == "full" else "metadata",
        "Prompts and tool data may leave this machine",
    )
    CONSOLE.print(Panel(table, title="Setup", border_style="fugue.gold"))
    CONSOLE.print(f"Agents: [link={status.links.agents}]{status.links.agents}[/link]")


def _print_checks(checks: Any) -> None:
    table = Table("Check", "State", "Detail", box=box.SIMPLE_HEAD)
    for check in checks:
        table.add_row(check.name, _state(check.ok), check.detail)
    CONSOLE.print(Panel(table, title="Preflight", border_style="fugue.gold"))


def _print_context_preparation(records: Any) -> None:
    table = Table("System", "Task", "State", "Detail", box=box.SIMPLE_HEAD)
    for record in records:
        table.add_row(
            record.system_id,
            record.task_id,
            record.status,
            record.detail,
        )
    if records:
        CONSOLE.print(Panel(table, title="Context preparation", border_style="fugue.cyan"))
    else:
        CONSOLE.print("[fugue.muted]No context artifacts were required.[/]")


def _run_command(args: argparse.Namespace) -> int:
    if args.run_id:
        return _run_worker(args)
    service = OperatorService(args.repo_root, args.env_file)
    experiment = _load_experiment_arg(args)
    request = _request_from_args(args, experiment.id)
    inline_experiment = bool(args.experiment_file or args.manifest or not args.experiment)
    if args.preview:
        preview = (
            service.preview_experiment(experiment, request=request)
            if inline_experiment
            else service.preview(request)
        )
        if args.json:
            from fugue.bench.operator import as_json

            print(as_json(preview))
        else:
            _print_preview(preview)
        return 0
    run = service.launch(request, experiment=experiment if inline_experiment else None)
    if args.json:
        from fugue.bench.operator import as_json

        final = run if args.detach else service.wait_for_run(run.run_id)
        print(as_json(final))
        return 0 if final.status in {"starting", "running", "passed"} else 1
    if args.detach:
        _print_started_run(run)
        return 0
    return _wait_for_run(service, run.run_id)


def _run_worker(args: argparse.Namespace) -> int:
    run_id = getattr(args, "run_id", None) or new_run_id()
    experiment = _load_experiment_arg(args)
    service = OperatorService(args.repo_root, args.env_file)
    request = _request_from_args(args, experiment.id)
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
        preparation = service.prepare_context(
            request,
            experiment=experiment,
            run_id=run_id,
        )
        _print_context_preparation(preparation)
        rendered = service.rendered_jobs(request, run_id=run_id, experiment=experiment)
        for job in rendered:
            if not job.applicable:
                CONSOLE.print(f"[yellow]skip[/] {job.job_name}: {job.skip_reason}")
                continue
            CONSOLE.print("[cyan]+[/] " + " ".join(shlex.quote(part) for part in job.command))
            print(f"# config: {job.config_path}")
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
        live_evaluations = None
        observability_error = None
        run_env = rendered[0].env if rendered else load_env(args.env_file)
        if (
            cells
            and run_env.get("WANDB_API_KEY", "").strip()
            and run_env.get("FUGUE_DISABLE_LIVE_EVALUATIONS", "").lower()
            not in {"1", "true", "yes"}
        ):
            try:
                live_evaluations = LiveEvaluationCoordinator(
                    cells,
                    repo_root=args.repo_root,
                    project=trace_project_slug(run_env),
                    env=run_env,
                )
            except Exception as exc:
                observability_error = f"{type(exc).__name__}: {exc}"
                CONSOLE.print(
                    "[yellow]Weave live evaluation unavailable:[/] "
                    f"{observability_error}"
                )
        outcomes = execute_cells(
            cells,
            repo_root=args.repo_root,
            max_workers=concurrency,
            cell_started=(
                live_evaluations.begin_cell if live_evaluations is not None else None
            ),
            cell_finished=(
                live_evaluations.finish_cell if live_evaluations is not None else None
            ),
        )
        publication = live_evaluations.finalize() if live_evaluations else None
        publication_failures = list(publication.failures if publication else ())
        if observability_error:
            publication_failures.insert(0, observability_error)
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
                "observability_status": (
                    "failed" if publication_failures else "passed"
                ),
                "evaluation_runs": (
                    [
                        {
                            "candidate_id": item.candidate_id,
                            "name": item.name,
                            "url": item.url,
                            "evaluation_ref": item.evaluation_ref,
                            "model_ref": item.model_ref,
                            "linked_predictions": item.linked_predictions,
                        }
                        for item in publication.evaluations
                    ]
                    if publication
                    else []
                ),
                "evaluation_failures": publication_failures,
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


def _request_from_args(
    args: argparse.Namespace,
    experiment_id: str,
) -> ExperimentRequest:
    return ExperimentRequest(
        experiment_id=experiment_id,
        manifest=getattr(args, "manifest", None),
        preset=getattr(args, "preset", None),
        workloads=tuple(_csv(getattr(args, "workloads", None)) or []),
        harnesses=tuple(_csv(getattr(args, "harnesses", None)) or []),
        systems=tuple(_csv(getattr(args, "systems", None)) or []),
        variants=tuple(_csv(getattr(args, "variants", None)) or []),
        model=getattr(args, "model", None),
        builder_model=getattr(args, "builder_model", None),
        judge_model=getattr(args, "judge_model", None),
        n_attempts=getattr(args, "n_attempts", None),
        n_tasks=getattr(args, "n_tasks", None),
        n_concurrent=getattr(args, "n_concurrent", None),
        run_name=getattr(args, "run_name", None),
        tags=tuple(_csv(getattr(args, "tags", None)) or []),
        jobs_dir=getattr(args, "jobs_dir", None),
        trace_content=getattr(args, "trace_content", None),
    )


def _tui(args: argparse.Namespace) -> int:
    from fugue.tui import run_tui

    screen = "compose" if args.screen == "plan" else args.screen
    run_tui(initial_screen=screen, experiment_id=args.experiment)
    return 0


def _plan(args: argparse.Namespace) -> int:
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
    save_id = args.save
    run_requested = args.run
    open_tui = False
    draft_shown = False
    if CONSOLE.is_terminal and not args.json and not save_id and not run_requested:
        _print_draft(draft)
        draft_shown = True
        action = Prompt.ask(
            "Next",
            choices=("tui", "save", "run", "both", "discard"),
            default="discard",
        )
        open_tui = action == "tui"
        run_requested = action in {"run", "both"}
        if action in {"save", "both"}:
            save_id = Prompt.ask("Experiment id", default=f"{draft.experiment.id}-planned")
    saved = (
        composer.save(
            draft,
            experiment_id=save_id,
            replace_assets=args.replace_assets,
        )
        if save_id
        else None
    )
    if open_tui:
        from fugue.tui import run_tui

        run_tui(
            initial_screen="compose",
            experiment_id=args.base_experiment,
            service=service,
            initial_draft=draft,
        )
        return 0
    run = None
    if run_requested:
        if draft.experiment.trace_content == "full" and CONSOLE.is_terminal and not args.yes:
            if not Confirm.ask("Run with full prompt, response, and tool content in Weave?"):
                run_requested = False
        if not run_requested:
            return 0
        saved = composer.save(
            draft, experiment_id=save_id, replace_assets=args.replace_assets
        ) if save_id and saved is None else saved
        if draft.assets and not saved:
            raise ValueError(
                "save the experiment and its proposed prompt or skill before running"
            )
        selected = saved or draft.experiment
        run = service.launch(
            ExperimentRequest(experiment_id=selected.id),
            experiment=None if saved else selected,
        )
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
    if not draft_shown:
        _print_draft(draft)
    if saved:
        CONSOLE.print(f"[green]saved[/] configs/fugue/experiments/{saved.id}.yaml")
    if run:
        _print_started_run(run)
    elif not saved:
        CONSOLE.print("[dim]Draft only. Use --save ID or --run to accept it.[/]")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    from fugue.bench.ai import (
        ExperimentAnalyst,
        get_analysis,
        list_analyses,
        save_analysis,
    )
    from fugue.bench.operator import OperatorService, as_json

    if args.list_saved:
        values = list_analyses(args.repo_root)
        if args.json:
            print(json.dumps(values, indent=2, sort_keys=True))
        else:
            table = Table("ID", "Title", box=box.SIMPLE_HEAD)
            for item in values:
                table.add_row(item["id"], item["title"])
            CONSOLE.print(table if values else "[fugue.muted]No saved analyses.[/]")
        return 0
    service = OperatorService(args.repo_root, args.env_file)
    analyst = ExperimentAnalyst(service)
    if args.saved:
        spec = get_analysis(args.saved, args.repo_root)
        if args.source:
            spec = replace(spec, source=args.source)
    else:
        question = " ".join(args.question).strip()
        if not question:
            raise ValueError("analysis question is required unless --saved or --list is used")
        spec = asyncio.run(
            analyst.plan(
                question,
                filters=_key_value_args(args.filter),
                model=args.model,
                source=args.source,
            )
        )
    preview = analyst.prepare(spec)
    if args.save:
        save_analysis(replace(spec, id=args.save), args.repo_root)
    if not args.json:
        _print_analysis_preview(preview)
    execute = args.yes or (
        CONSOLE.is_terminal and Confirm.ask("Generate the evidence-backed report?")
    )
    if not execute:
        if args.json:
            print(as_json(preview))
        else:
            CONSOLE.print("[fugue.muted]Scope only. Use --yes to generate the report.[/]")
        return 0
    result = asyncio.run(analyst.execute(preview, model=args.model))
    if args.json:
        print(as_json(result))
    else:
        CONSOLE.print(Panel(Markdown(result.report), title="Analysis", border_style="fugue.cyan"))
        CONSOLE.print(f"Report: [fugue.cyan]{result.report_dir / 'report.md'}[/]")
    return 0


def _setup(args: argparse.Namespace) -> int:
    from fugue.bench.operator import as_json

    service = OperatorService(args.repo_root, args.env_file)
    request = ExperimentRequest(
        experiment_id=args.experiment,
        manifest=args.manifest,
        preset=args.preset,
        workloads=tuple(_csv(args.workloads) or []),
        systems=tuple(_csv(args.systems) or []),
        model=args.model,
        builder_model=args.builder_model,
        judge_model=args.judge_model,
        trace_content=args.trace_content,
    )
    if args.check:
        checks = service.preflight(request, live=True)
        if args.json:
            print(as_json(checks))
        else:
            _print_checks(checks)
        return 0 if all(check.ok for check in checks) else 1
    if args.start_bridge:
        files = service.start_bridge(request)
        if args.json:
            print(as_json(files))
        else:
            CONSOLE.print(
                Panel(
                    f"Bridge is running from [fugue.cyan]{files.runtime_dir}[/]",
                    title="Bridge",
                    border_style="fugue.success",
                )
            )
        return 0
    if args.prepare_context:
        records = service.prepare_context(request, rebuild=args.rebuild)
        if args.json:
            print(as_json(records))
        else:
            _print_context_preparation(records)
        return 0
    status = service.status(request)
    if args.json:
        print(as_json(status))
    else:
        _print_setup(status)
    return 0 if all(route.key_present for route in status.routes) and status.trace_key_present else 1


def _runs(args: argparse.Namespace) -> int:
    from fugue.bench.operator import OperatorService, as_json

    service = OperatorService(args.repo_root, args.env_file)
    if not args.run_id:
        if any((args.logs, args.cancel, args.export_run, args.open)):
            raise ValueError("a run id is required for this action")
        runs = service.runs()[: args.limit]
        if args.json:
            print(as_json(runs))
            return 0
        table = Table(title="Recent runs", box=box.SIMPLE_HEAD)
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
            CONSOLE.print("[fugue.muted]No runs yet. Start one with `fugue run pilot`.[/]")
        return 0
    if args.logs:
        if args.follow:
            try:
                for chunk in service.supervisor.follow_log(args.run_id, cell_id=args.cell):
                    print(chunk, end="", flush=True)
            except KeyboardInterrupt:
                return 130
        else:
            print(service.supervisor.read_log(args.run_id, cell_id=args.cell), end="")
        return 0
    if args.cancel:
        run = service.supervisor.cancel(args.run_id)
        if args.json:
            print(as_json(service.run_summary(args.run_id)))
        else:
            CONSOLE.print(f"{run.run_id}: {_status_markup(run.status)}")
        return 0
    if args.export_run:
        summary = service.export_run(
            args.run_id,
            out=args.out,
            fetch_weave=args.fetch_weave,
            to_weave=args.to_weave,
            republish=args.republish,
        )
        if args.json:
            print(as_json(summary))
        else:
            if summary.published:
                CONSOLE.print(
                    f"Published {summary.published} finalized candidate evaluation(s)"
                )
                for evaluation in summary.evaluations:
                    suffix = f" [cyan]{evaluation.url}[/]" if evaluation.url else ""
                    CONSOLE.print(
                        f"  {evaluation.name} ({evaluation.examples} examples){suffix}"
                    )
            if summary.skipped:
                CONSOLE.print(f"Skipped {summary.skipped} published candidate(s)")
            for failure in summary.publication_failures:
                CONSOLE.print(f"[red]Publication failed:[/] {failure}")
            CONSOLE.print(f"Exported {summary.rows} rows to [cyan]{summary.path}[/]")
        return 0
    if args.open:
        links = service.run_links(args.run_id)
        url = links.project if args.open == "project" else links.agents
        conversation_id = None
        if args.open == "trace":
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
        if args.json:
            print(as_json({"url": url, "conversation_id": conversation_id}))
        elif args.print_only:
            print(url)
        else:
            webbrowser.open(url)
            CONSOLE.print(f"Opened [link={url}]{url}[/link]")
        if conversation_id and not args.json:
            CONSOLE.print(f"Conversation: [cyan]{conversation_id}[/]")
        return 0
    run = service.run_summary(args.run_id)
    if args.json:
        print(as_json(run))
    else:
        CONSOLE.print(_run_panel(run))
        CONSOLE.print(_cells_table(run))
    return 0


def _print_started_run(run: Any) -> None:
    CONSOLE.print(
        Panel(
            f"[fugue.success]started[/] [bold]{run.run_id}[/]\n"
            f"Logs: [fugue.cyan]{run.log_path}[/]\n"
            f"Follow: [bold]fugue runs {run.run_id} --logs --follow[/]",
            title=run.run_name,
            border_style="fugue.success",
        )
    )


def _run_panel(run: Any) -> Panel:
    return Panel(
        f"{_status_markup(run.status)}  "
        f"[fugue.success]{run.passed} passed[/]  "
        f"[fugue.coral]{run.failed} failed[/]  "
        f"{run.pending} pending  {run.not_applicable} not applicable",
        title=f"{run.run_name}  [{run.run_id}]",
        border_style="fugue.cyan" if run.status in {"starting", "running"} else "fugue.gold",
    )


def _cells_table(run: Any) -> Table:
    table = Table("Harness", "Variant", "Context", "Task", "Status", box=box.SIMPLE_HEAD)
    for cell in run.cells:
        table.add_row(
            cell.harness,
            cell.variant_id,
            cell.context_system_id,
            cell.task_id,
            _status_markup(cell.status),
        )
    if not run.cells:
        table.add_row("-", "-", "-", "waiting for planner", _status_markup(run.status))
    return table


def _wait_for_run(service: OperatorService, run_id: str) -> int:
    terminal = {"passed", "failed", "cancelled", "interrupted"}
    if not CONSOLE.is_terminal:
        try:
            for chunk in service.supervisor.follow_log(run_id):
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            service.supervisor.cancel(run_id)
            return 130
        run = service.run_summary(run_id)
        return 0 if run.status == "passed" else 1
    offset = 0
    log_tail = ""
    try:
        with Live(console=CONSOLE, refresh_per_second=4) as live:
            while True:
                run = service.run_summary(run_id)
                chunk, offset = service.supervisor.read_log_chunk(run_id, offset=offset)
                if chunk:
                    log_tail = (log_tail + chunk)[-8_000:]
                live.update(
                    Group(
                        _run_panel(run),
                        _cells_table(run),
                        Panel(log_tail or "Waiting for output...", title="Live log", border_style="fugue.muted"),
                    )
                )
                if run.status in terminal:
                    return 0 if run.status == "passed" else 1
                time.sleep(0.25)
    except KeyboardInterrupt:
        service.supervisor.cancel(run_id)
        return 130


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


def _now() -> str:
    return datetime.now(UTC).isoformat()


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


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    raise SystemExit(main())
