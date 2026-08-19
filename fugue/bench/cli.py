from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import webbrowser
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from importlib.resources import as_file, files
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
from fugue.bench.execution import new_run_id
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
from fugue.bench.templates import standalone_template_ids
from fugue.bench.workloads import (
    load_workload_dataset,
    run_retrieval_workload,
    run_sequence_workload,
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

_STANDALONE_STUDY_MARKER = ".fugue-study.json"
_STANDALONE_TEMPLATE_IDS = standalone_template_ids()


class FugueArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", RichHelpFormatter)
        super().__init__(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ["_context-evaluate"]:
        return _internal_context_evaluate(raw_argv[1:])
    raw_argv = _normalize_runs_argv(raw_argv)
    parser = _parser()
    args = parser.parse_args(raw_argv)
    if args.command is None:
        return _command_center(parser)
    return int(args.handler(args))


def _internal_context_evaluate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--variant", required=True)
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
        description=(
            "Plan, run, and analyze isolated Agent experiments with canonical "
            "local evidence and optional W&B/Weave publication."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    doctor = subparsers.add_parser(
        "doctor", help="Inspect this Fugue installation and optional capabilities"
    )
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--workspace", type=Path, default=Path.cwd())
    doctor.add_argument("--model")
    doctor.add_argument(
        "--require",
        action="append",
        choices=("local-runner",),
        default=[],
        metavar="CAPABILITY",
        help=(
            "Fail unless the requested capability is ready; repeatable. "
            "local-runner checks Python, architecture, Harbor, Docker, assets, "
            "and the selected model credential."
        ),
    )
    doctor.add_argument(
        "--env-file",
        type=Path,
        help="Read the selected model credential from a dotenv file",
    )
    doctor.set_defaults(handler=_doctor)

    init = subparsers.add_parser(
        "init", help="Scaffold a standalone Agent-change comparison"
    )
    init.add_argument(
        "--template",
        choices=_STANDALONE_TEMPLATE_IDS,
        default="prompt-change",
    )
    init.add_argument("destination", nargs="?", type=Path, default=Path("comparison"))
    init.add_argument("--force", action="store_true")
    init.set_defaults(handler=_comparison_init)

    check = subparsers.add_parser(
        "check", help="Validate authored comparison readiness without model spend"
    )
    check.add_argument("comparison", type=Path)
    _add_common_args(check, json_output=True, infer_comparison_root=True)
    check.set_defaults(handler=_comparison_check)

    compare = subparsers.add_parser(
        "compare", help="Preview or run one baseline-versus-candidate comparison"
    )
    compare.add_argument("comparison", type=Path)
    action = compare.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--prepare",
        action="store_true",
        help="Freeze inputs and build the exact local runtimes without running cells",
    )
    action.add_argument("--preview", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument(
        "--resume",
        metavar="RUN_ID",
        help=(
            "Resume one immutable run without rerunning valid cells; requires "
            "a current --approval receipt for the same exact preview. The "
            "run retains its original immutable approval and stage locks."
        ),
    )
    compare.add_argument("--approval")
    compare.add_argument(
        "--fetch-weave",
        action="store_true",
        help=(
            "For weave_required studies, fetch hosted Weave evidence during "
            "finalization. In local mode this flag does not trigger hosted "
            "evidence hydration."
        ),
    )
    _add_common_args(compare, json_output=True, infer_comparison_root=True)
    compare.set_defaults(handler=_comparison_compare)

    approve = subparsers.add_parser(
        "approve", help="Approve one exact comparison preview"
    )
    approve.add_argument("preview_digest")
    approve.add_argument("--max-usd", type=float, required=True)
    approve.add_argument("--max-cells", type=int)
    approve.add_argument("--approved-by", default="operator")
    approve.add_argument("--expires-in", type=int, default=3600)
    approve.add_argument("--operation-id")
    approve.add_argument("--repo-root", type=Path, default=Path.cwd())
    approve.set_defaults(handler=_comparison_approve)

    result = subparsers.add_parser("result", help="Read an exported comparison result")
    result.add_argument("comparison", nargs="?", default="latest")
    result.add_argument("--json", action="store_true")
    result.add_argument("--open", action="store_true", dest="open_result")
    result_action = result.add_mutually_exclusive_group()
    result_action.add_argument("--append-invalidation", type=Path)
    result_action.add_argument(
        "--authorize-followup",
        type=Path,
        metavar="COMPARISON",
        help=(
            "Bind this exact reviewed V3 result as the declared prerequisite "
            "for a follow-up comparison"
        ),
    )
    result_action.add_argument(
        "--signoff-by",
        help=(
            "Attach release-owner actionability sign-off to a "
            "ready_for_signoff V3 result"
        ),
    )
    result.add_argument("--reviewed-by")
    result.add_argument("--reviewed-at")
    result.add_argument("--research-id")
    result.add_argument("--repo-root", type=Path, default=Path.cwd())
    result.set_defaults(handler=_comparison_result)

    study = subparsers.add_parser(
        "study",
        help="Build a digest-bound local Research index (not a native W&B Study)",
    )
    study_actions = study.add_subparsers(
        dest="study_action", metavar="ACTION", required=True
    )
    study_index = study_actions.add_parser(
        "index",
        help=("Index exact local results and their scoped Weave publication receipts"),
    )
    study_index.add_argument("--research-id", required=True)
    study_index.add_argument("--title", required=True)
    study_index.add_argument("--objective", required=True)
    study_index.add_argument(
        "--source",
        action="append",
        nargs=2,
        required=True,
        metavar=("RESULT", "WEAVE_RECEIPT"),
        help=(
            "Add one exact ComparisonResultV3 and its scoped Weave publication "
            "receipt; repeat for each Study."
        ),
    )
    study_index.add_argument("--output", type=Path)
    _add_common_args(study_index, json_output=True)
    study_index.set_defaults(handler=_study_index)

    publish_command = subparsers.add_parser(
        "publish",
        help=(
            "Publish digest-bound projections of local results or Research "
            "indexes to an optional W&B backend"
        ),
    )
    publish_actions = publish_command.add_subparsers(
        dest="publish_action", metavar="BACKEND", required=True
    )
    publish_weave = publish_actions.add_parser(
        "weave",
        help=(
            "Publish a sanitized, digest-bound projection of a verified local "
            "result to W&B Weave"
        ),
        description=(
            "Publish sanitized result and evidence-chain projections to W&B Weave. "
            "Raw local transcript and tool-event artifact files remain local, and "
            "the canonical local result digest does not change. Before you publish, "
            "inspect each sanitized_answer_excerpt and the evidence metadata in "
            "result.json for sensitive data."
        ),
    )
    publish_weave.add_argument("result", type=Path)
    publish_weave.add_argument("--project", required=True, metavar="ENTITY/PROJECT")
    publish_weave.add_argument(
        "--research-id",
        help=(
            "Label this Weave publication with a Fugue Research scope. If you "
            "omit --study-id, Fugue uses the result comparison id as the Study "
            "key. This option does not create a Study Console Research record."
        ),
    )
    publish_weave.add_argument(
        "--study-id",
        help=(
            "Stable Fugue Study key within --research-id. This key is not a "
            "generated Study Console ID."
        ),
    )
    publish_weave.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    publish_weave.add_argument("--receipt", type=Path)
    _add_common_args(publish_weave, json_output=True)
    publish_weave.set_defaults(handler=_publish_local_result)
    publish_index = publish_actions.add_parser(
        "wandb-index",
        help=(
            "Publish a W&B index Run and an immutable Artifact version from an "
            "unchanged Research index"
        ),
        description=(
            "Publish a deterministic W&B Run whose Table lists the indexed Studies. "
            "Also publish an immutable Artifact version that contains the unchanged "
            "Research index. This command does not create a Study Console projection "
            "or a W&B Report."
        ),
    )
    publish_index.add_argument("index", type=Path)
    publish_index.add_argument("--project", required=True, metavar="ENTITY/PROJECT")
    publish_index.add_argument("--receipt", type=Path)
    _add_common_args(publish_index, json_output=True)
    publish_index.set_defaults(handler=_publish_research_index)
    publish_report = publish_actions.add_parser(
        "wandb-report",
        help=("Publish a verified W&B Report view of one Research index"),
        description=(
            "Publish a W&B Report that presents one Research index. Fugue reads the "
            "saved Report from W&B and compares its fields and Markdown with the "
            "prepared projection. The local Research index and index-publication "
            "receipt remain authoritative. The Reports API is Public Preview. This "
            "command does not change access settings or request a public share link."
        ),
    )
    publish_report.add_argument("index", type=Path)
    publish_report.add_argument(
        "--index-receipt",
        required=True,
        type=Path,
        help=(
            "Immutable receipt from fugue publish wandb-index; the Report uses the "
            "W&B project and application origin bound by this receipt"
        ),
    )
    publish_report.add_argument("--receipt", type=Path)
    _add_common_args(publish_report, json_output=True)
    publish_report.set_defaults(handler=_publish_research_report)

    demo = subparsers.add_parser(
        "demo", help="Run a deterministic no-key comparison replay"
    )
    demo.add_argument("demo", choices=("source-use",))
    demo.add_argument("--out", type=Path)
    demo.add_argument("--json", action="store_true")
    demo.add_argument("--repo-root", type=Path, default=Path.cwd())
    demo.set_defaults(handler=_comparison_demo)

    sandbox = subparsers.add_parser(
        "sandbox", help="Build and qualify remote Sandbox runtimes"
    )
    sandbox_backends = sandbox.add_subparsers(
        dest="sandbox_backend", metavar="BACKEND", required=True
    )
    wandb_sandbox = sandbox_backends.add_parser(
        "wandb", help="Qualify W&B Serverless Sandbox execution"
    )
    wandb_actions = wandb_sandbox.add_subparsers(
        dest="wandb_action", metavar="ACTION", required=True
    )
    wandb_build = wandb_actions.add_parser(
        "build-runtime",
        help="Build, scan, publish, and attest public Agent runtime images",
    )
    wandb_build.add_argument(
        "--comparison",
        type=Path,
        action="append",
        required=True,
        dest="comparisons",
    )
    wandb_build.add_argument("--platform", default="linux/amd64")
    wandb_build.add_argument("--image", required=True)
    wandb_build.add_argument("--push", action="store_true")
    wandb_build.add_argument("--output-manifest", type=Path, required=True)
    wandb_build.add_argument("--sbom-dir", type=Path)
    wandb_build.add_argument("--repo-root", type=Path, default=Path.cwd())
    wandb_build.set_defaults(handler=_wandb_sandbox)
    wandb_lock = wandb_actions.add_parser(
        "lock-runtime",
        help="Accept one published, anonymously pullable runtime manifest",
    )
    wandb_lock.add_argument("--manifest", type=Path, required=True)
    wandb_lock.add_argument(
        "--output",
        type=Path,
        default=Path(".fugue/wandb-serverless-runtime.lock.json"),
    )
    wandb_lock.set_defaults(handler=_wandb_sandbox)
    wandb_doctor_parser = wandb_actions.add_parser(
        "doctor",
        help="Create, probe, delete, and verify one disposable W&B Sandbox",
    )
    wandb_doctor_parser.add_argument(
        "--lock",
        type=Path,
        default=Path(".fugue/wandb-serverless-runtime.lock.json"),
    )
    wandb_doctor_parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
    )
    wandb_doctor_parser.set_defaults(handler=_wandb_sandbox)

    provider = subparsers.add_parser(
        "provider",
        help="Validate and lock a language-neutral evaluation provider",
    )
    provider_actions = provider.add_subparsers(
        dest="provider_action", metavar="ACTION", required=True
    )
    provider_validate = provider_actions.add_parser(
        "validate", help="Validate a provider descriptor without writing state"
    )
    provider_validate.add_argument("--command", required=True)
    provider_validate.add_argument("--timeout", type=float, default=30.0)
    provider_validate.set_defaults(handler=_component_provider)
    provider_lock = provider_actions.add_parser(
        "lock", help="Lock a provider command, descriptor, source, and executable"
    )
    provider_lock.add_argument("--command", required=True)
    provider_lock.add_argument("--output", type=Path, required=True)
    provider_lock.add_argument("--timeout", type=float, default=30.0)
    provider_lock.set_defaults(handler=_component_provider)
    provider_schema = provider_actions.add_parser(
        "schema", help="Write strict Provider Contract V1 JSON Schemas"
    )
    provider_schema.add_argument(
        "--destination", type=Path, default=Path("schemas/fugue/providers")
    )
    provider_schema.set_defaults(handler=_component_provider)
    provider_scaffold = provider_actions.add_parser(
        "scaffold", help="Create a dependency-free Evaluation Provider V1 scaffold"
    )
    provider_scaffold.add_argument("destination", type=Path)
    provider_scaffold.add_argument("--provider-id", required=True)
    provider_scaffold.add_argument("--force", action="store_true")
    provider_scaffold.set_defaults(handler=_component_provider)
    provider_conformance_parser = provider_actions.add_parser(
        "conformance",
        help="Validate provider artifacts and lifecycle without running an experiment",
    )
    provider_conformance_parser.add_argument("--provider", type=Path, required=True)
    provider_conformance_parser.add_argument("--candidate", required=True)
    provider_conformance_parser.add_argument("--suite", required=True)
    provider_conformance_parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Inspect only this exact provider task id; repeat to select multiple tasks.",
    )
    provider_conformance_parser.add_argument(
        "--exercise-run-cell",
        action="store_true",
        help="Exercise credential-free run-cell protocol operations locally",
    )
    provider_conformance_parser.add_argument("--output", type=Path)
    provider_conformance_parser.add_argument("--timeout", type=float, default=120.0)
    provider_conformance_parser.set_defaults(handler=_component_provider)
    taskset = subparsers.add_parser(
        "taskset", help="Build or import simple Agent evaluation tasksets"
    )
    taskset_actions = taskset.add_subparsers(
        dest="taskset_action", metavar="ACTION", required=True
    )
    taskset_schema = taskset_actions.add_parser(
        "schema", help="Write the public-task and private-label JSON Schemas"
    )
    taskset_schema.add_argument(
        "--destination", type=Path, default=Path("schemas/fugue")
    )
    taskset_schema.set_defaults(handler=_component_taskset)
    taskset_weave = taskset_actions.add_parser(
        "import-weave",
        help="Import one immutable Weave Dataset as a public taskset",
    )
    taskset_weave.add_argument("--dataset", required=True)
    taskset_weave.add_argument("--as", dest="import_id", required=True)
    taskset_weave.add_argument("--env-file", type=Path, default=Path(".env"))
    taskset_weave.add_argument("--repo-root", type=Path, default=Path.cwd())
    taskset_weave.set_defaults(handler=_component_taskset)
    mcp_component = subparsers.add_parser(
        "mcp", help="Import, inspect, and lock a normal MCP server declaration"
    )
    mcp_actions = mcp_component.add_subparsers(
        dest="mcp_action", metavar="ACTION", required=True
    )
    mcp_prepare_wandb = mcp_actions.add_parser(
        "prepare-wandb-release",
        help=(
            "Resolve and freeze the W&B MCP staging/0.4.0 reference study "
            "without running Agent cells"
        ),
    )
    mcp_prepare_wandb.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Workspace that will receive the immutable reference-study bundle",
    )
    mcp_prepare_wandb.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Credential file recorded only as an operator input; values are "
            "never copied into the source lock"
        ),
    )
    mcp_prepare_wandb.add_argument(
        "--platform",
        choices=("linux/amd64", "linux/arm64"),
        default="linux/amd64",
        help="Target platform for the immutable MCP runtime locks",
    )
    mcp_prepare_wandb.set_defaults(handler=_component_mcp)
    mcp_import = mcp_actions.add_parser(
        "import", help="Import one selected server from Codex TOML or mcpServers JSON"
    )
    mcp_import.add_argument("--config", type=Path, required=True)
    mcp_import.add_argument("--server", required=True)
    mcp_import.add_argument("--as", dest="import_id", required=True)
    mcp_import.add_argument("--allow-host", action="append", default=[])
    mcp_import.add_argument("--repo-root", type=Path, default=Path.cwd())
    mcp_import.set_defaults(handler=_component_mcp)
    mcp_add = mcp_actions.add_parser(
        "add", help="Record one explicit argv-based stdio MCP declaration"
    )
    mcp_add.add_argument("import_id")
    mcp_add.add_argument("--required-env", action="append", default=[])
    mcp_add.add_argument("--allow-host", action="append", default=[])
    mcp_add.add_argument("--repo-root", type=Path, default=Path.cwd())
    mcp_add.add_argument("argv", nargs=argparse.REMAINDER)
    mcp_add.set_defaults(handler=_component_mcp)
    for action_name in ("inspect", "lock"):
        action_parser = mcp_actions.add_parser(
            action_name, help=f"{action_name.title()} one imported MCP declaration"
        )
        action_parser.add_argument("import_id")
        action_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        if action_name == "lock":
            action_parser.add_argument(
                "--acknowledge-package-code", action="store_true"
            )
            action_parser.add_argument(
                "--platform",
                choices=("linux/amd64", "linux/arm64"),
            )
        action_parser.set_defaults(handler=_component_mcp)

    skills_component = subparsers.add_parser(
        "skills", help="Import, inspect, and lock a standard Agent Skill"
    )
    skill_actions = skills_component.add_subparsers(
        dest="skills_action", metavar="ACTION", required=True
    )
    skill_import = skill_actions.add_parser(
        "import", help="Import a local Skill folder or exact Git Skill source"
    )
    skill_import.add_argument("source")
    skill_import.add_argument("--as", dest="import_id")
    skill_import.add_argument("--repo-root", type=Path, default=Path.cwd())
    skill_import.set_defaults(handler=_component_skills)
    for action_name in ("inspect", "lock"):
        action_parser = skill_actions.add_parser(
            action_name, help=f"{action_name.title()} one imported Agent Skill"
        )
        action_parser.add_argument("skill_id")
        action_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        action_parser.set_defaults(handler=_component_skills)

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
    run.add_argument(
        "experiment", nargs="?", help="Saved experiment id (default: pilot)"
    )
    _add_run_args(run)
    run.add_argument(
        "--preview",
        action="store_true",
        help="Show the matrix without writing runtime state",
    )
    run.add_argument(
        "--detach",
        action="store_true",
        help="Start the durable run and return immediately",
    )
    run.add_argument(
        "--json",
        action="store_true",
        help="Emit structured output without Rich decoration",
    )
    run.add_argument("--run-id", help=argparse.SUPPRESS)
    run.add_argument("--experiment-file", type=Path, help=argparse.SUPPRESS)
    run.set_defaults(handler=_run_command)

    runs = subparsers.add_parser("runs", help="Inspect and manage durable runs")
    runs.add_argument("--run-id", help=argparse.SUPPRESS)
    runs.add_argument(
        "--limit",
        type=_positive_cli_int,
        default=20,
        help="Maximum recent runs to list",
    )
    _add_common_args(runs, json_output=True)
    run_actions = runs.add_subparsers(dest="runs_action", metavar="ACTION")
    logs = run_actions.add_parser("logs", help="Read run or selected-cell logs")
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--cell")
    _add_common_args(logs, json_output=True)
    cancel = run_actions.add_parser("cancel", help="Cancel the managed process group")
    _add_common_args(cancel, json_output=True)
    export_run = run_actions.add_parser("export", help="Write normalized JSONL")
    export_run.add_argument("--out", type=Path)
    export_run.add_argument("--fetch-weave", action="store_true")
    export_run.add_argument("--to-weave", action="store_true")
    export_run.add_argument("--republish", action="store_true")
    export_run.add_argument("--republish-reason")
    _add_common_args(export_run, json_output=True)
    package = run_actions.add_parser("package", help="Package one candidate")
    package.add_argument("candidate")
    package.add_argument("--workspace", type=Path, required=True)
    package.add_argument("--image", required=True)
    package.add_argument("--platform", default="linux/amd64")
    package.add_argument("--allow-failed", action="store_true")
    package.add_argument("--yes", action="store_true")
    _add_common_args(package, json_output=True)
    open_run = run_actions.add_parser("open", help="Open a W&B destination")
    open_run.add_argument(
        "destination", choices=("agents", "evaluation", "trace", "project")
    )
    open_run.add_argument("--cell")
    open_run.add_argument("--print", action="store_true", dest="print_only")
    _add_common_args(open_run, json_output=True)
    runs.set_defaults(handler=_runs)

    analyze = subparsers.add_parser(
        "analyze", help="Analyze experiment results with Fugue AI"
    )
    analyze.add_argument("question", nargs="*")
    source = analyze.add_mutually_exclusive_group()
    source.add_argument("--saved", help="Run a saved analysis definition")
    source.add_argument(
        "--list",
        action="store_true",
        dest="list_saved",
        help="List saved analysis definitions",
    )
    analyze.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Required FIELD=VALUE scope filter",
    )
    analyze.add_argument("--model", help="Analyst model route")
    analyze.add_argument(
        "--source",
        choices=("local", "hybrid"),
        help="Use local outcomes or narrow Weave enrichment",
    )
    analyze.add_argument("--save", help="Save the generated analysis definition")
    analyze.add_argument(
        "--yes", action="store_true", help="Confirm report generation without prompting"
    )
    _add_common_args(analyze, json_output=True)
    analyze.set_defaults(handler=_analyze)

    setup = subparsers.add_parser(
        "setup", help="Inspect and prepare Fugue dependencies"
    )
    setup.add_argument("--experiment", default="pilot")
    setup.add_argument("--model")
    setup.add_argument("--builder-model")
    setup.add_argument("--judge-model")
    setup.add_argument("--preset")
    setup.add_argument("--manifest", type=Path)
    setup.add_argument("--workloads")
    setup.add_argument("--systems")
    setup.add_argument("--harnesses")
    setup.add_argument("--variants")
    setup.add_argument("-k", "--n-attempts", type=int)
    setup.add_argument("-n", "--n-concurrent", type=int)
    setup.add_argument("-l", "--n-tasks", type=int)
    setup.add_argument("--trace-content", choices=("full", "metadata"))
    operation = setup.add_mutually_exclusive_group()
    operation.add_argument(
        "--check", action="store_true", help="Run observational live preflight"
    )
    operation.add_argument(
        "--start-bridge", action="store_true", help="Start the local LiteLLM bridge"
    )
    operation.add_argument(
        "--start-services",
        action="store_true",
        help="Start selected managed context services",
    )
    operation.add_argument(
        "--service-status",
        action="store_true",
        help="Inspect selected managed context services without changing them",
    )
    operation.add_argument(
        "--stop-services",
        action="store_true",
        help="Stop selected managed context services and preserve their data",
    )
    operation.add_argument(
        "--prepare",
        action="store_true",
        help="Build all locked context and harness artifacts selected by the plan",
    )
    operation.add_argument(
        "--prepare-context",
        action="store_true",
        help="Build selected context artifacts",
    )
    operation.add_argument(
        "--skills",
        action="store_true",
        help="Fetch and inspect selected remote skills without executing repository code",
    )
    operation.add_argument(
        "--approve-skill",
        metavar="ID=DIGEST",
        help="Approve an inspected remote skill at exactly this sha256 digest",
    )
    setup.add_argument(
        "--rebuild", action="store_true", help="Ignore reusable context cache entries"
    )
    setup.add_argument(
        "--refresh-skills",
        action="store_true",
        help="Refetch pinned Git objects while inspecting remote skills",
    )
    setup.add_argument(
        "--acknowledge-risk",
        action="append",
        default=[],
        metavar="FINDING",
        help="Acknowledge a named review finding during skill approval",
    )
    _add_common_args(setup, json_output=True)
    setup.set_defaults(handler=_setup)

    tui = subparsers.add_parser("tui", help="Open the full-screen terminal workspace")
    tui.add_argument(
        "--screen", choices=("plan", "runs", "results", "setup"), default="plan"
    )
    tui.add_argument("--experiment", default="pilot")
    tui.set_defaults(handler=_tui)

    research = subparsers.add_parser(
        "research", help="Expose Fugue as a governed research substrate"
    )
    research_actions = research.add_subparsers(
        dest="research_action", metavar="ACTION", required=True
    )
    bootstrap = research_actions.add_parser(
        "bootstrap", help="Create local container state and secret files"
    )
    bootstrap.add_argument("--repo-root", type=Path, default=Path.cwd())
    bootstrap.add_argument("--wandb-api-key-file", type=Path)
    bootstrap.add_argument(
        "--trace-wandb-api-key-file",
        type=Path,
        help="Use a separate W&B credential for Weave evidence publication",
    )
    bootstrap.add_argument(
        "--env-file",
        type=Path,
        help="Read only allowlisted credentials from a dotenv file",
    )
    bootstrap.set_defaults(handler=_research)
    serve = research_actions.add_parser("serve", help="Run the typed HTTP and SSE API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--api-key")
    serve.add_argument("--env-file", type=Path, default=Path(".env"))
    serve.add_argument("--trace-sources", type=Path)
    serve.add_argument("--candidate-sources", type=Path)
    serve.add_argument("--repo-root", type=Path, default=Path.cwd())
    serve.set_defaults(handler=_research)
    mcp = research_actions.add_parser("mcp", help="Run the high-level MCP adapter")
    mcp.add_argument("--env-file", type=Path, default=Path(".env"))
    mcp.add_argument("--trace-sources", type=Path)
    mcp.add_argument("--candidate-sources", type=Path)
    mcp.add_argument("--repo-root", type=Path, default=Path.cwd())
    mcp.set_defaults(handler=_research)
    worker = research_actions.add_parser(
        "worker", help="Run the durable Harbor research worker"
    )
    worker.add_argument("--repo-root", type=Path, default=Path.cwd())
    worker.add_argument("--env-file", type=Path, default=Path(".env"))
    worker.add_argument("--candidate-sources", type=Path)
    worker.add_argument("--poll-interval", type=float, default=1.0)
    worker.add_argument("--once", action="store_true")
    worker.set_defaults(handler=_research)
    approve = research_actions.add_parser(
        "approve", help="Approve one exact preview outside the Agent interface"
    )
    approve.add_argument("preview_digest")
    approve.add_argument(
        "--subject-kind", choices=("experiment", "trace_audit"), default="experiment"
    )
    approve.add_argument("--max-usd", type=float, required=True)
    approve.add_argument("--max-cells", type=int)
    approve.add_argument("--approved-by", default="operator")
    approve.add_argument("--expires-in", type=int, default=3600)
    approve.add_argument("--operation-id")
    approve.add_argument("--repo-root", type=Path, default=Path.cwd())
    approve.set_defaults(handler=_research)
    publications = research_actions.add_parser(
        "publications",
        help="Inspect or replay public-safe research records",
    )
    publication_actions = publications.add_subparsers(
        dest="publication_action",
        metavar="ACTION",
        required=True,
    )
    replay_publications = publication_actions.add_parser(
        "replay",
        help="Republish immutable records to the configured projection sink",
    )
    replay_publications.add_argument("--repo-root", type=Path, default=Path.cwd())
    replay_publications.add_argument("--research-id")
    replay_publications.set_defaults(handler=_research)
    skill = research_actions.add_parser(
        "skill", help="Export the portable external-Agent skill"
    )
    skill_actions = skill.add_subparsers(
        dest="skill_action", metavar="ACTION", required=True
    )
    export_skill = skill_actions.add_parser("export", help="Export the packaged skill")
    export_skill.add_argument("--destination", type=Path, required=True)
    export_skill.set_defaults(handler=_research)
    return parser


def _research(args: argparse.Namespace) -> int:
    from fugue.research.runtime import load_secret_file_environment

    load_secret_file_environment()
    if args.research_action == "bootstrap":
        from fugue.research.bootstrap import bootstrap_container_secrets

        values = bootstrap_container_secrets(
            args.repo_root,
            wandb_api_key_file=args.wandb_api_key_file,
            trace_wandb_api_key_file=args.trace_wandb_api_key_file,
            env_file=args.env_file,
        )
        print(json.dumps(values, indent=2, sort_keys=True))
    elif args.research_action == "serve":
        from fugue.research.http import serve

        serve(
            args.repo_root,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            env_file=args.env_file,
            trace_sources=args.trace_sources,
            candidate_sources=args.candidate_sources,
        )
    elif args.research_action == "mcp":
        from fugue.research.candidate_sources import CandidateSourceRegistry
        from fugue.research.mcp import create_mcp_server
        from fugue.research.service import ResearchService
        from fugue.research.traces import TraceSourceRegistry

        registry = TraceSourceRegistry.from_file(args.trace_sources)
        candidates = CandidateSourceRegistry.from_file(args.candidate_sources)
        service = ResearchService(
            args.repo_root,
            args.env_file,
            trace_registry=registry,
            candidate_sources=candidates,
        )
        create_mcp_server(args.repo_root, service=service).run()
    elif args.research_action == "worker":
        from fugue.research.candidate_sources import CandidateSourceRegistry
        from fugue.research.service import ResearchService, ResearchWorker

        configured_candidates = args.candidate_sources
        if configured_candidates is None and os.getenv(
            "FUGUE_RESEARCH_CANDIDATE_SOURCES"
        ):
            configured_candidates = Path(os.environ["FUGUE_RESEARCH_CANDIDATE_SOURCES"])
        service = ResearchService(
            args.repo_root,
            args.env_file,
            candidate_sources=CandidateSourceRegistry.from_file(configured_candidates),
        )
        worker = ResearchWorker(service, poll_interval=args.poll_interval)
        if args.once:
            service.run_until_idle(worker.worker_id)
        else:
            try:
                worker.run_forever()
            except KeyboardInterrupt:
                pass
    elif args.research_action == "approve":
        from fugue.research.approvals import ApprovalLedger
        from fugue.research.store import StudyStore

        store = StudyStore(args.repo_root)
        operation_id = args.operation_id or f"approve-{args.preview_digest[:20]}"
        approval = ApprovalLedger(store.path).approve(
            subject_kind=args.subject_kind,
            preview_digest=args.preview_digest,
            maximum_cost_usd=args.max_usd,
            maximum_cells=args.max_cells,
            approved_by=args.approved_by,
            operation_id=operation_id,
            expires_in_seconds=args.expires_in,
        )
        print(json.dumps(approval.to_dict(), indent=2, sort_keys=True))
    elif args.research_action == "publications":
        from fugue.research.records import ResearchRecordPublisher
        from fugue.research.store import StudyStore

        store = StudyStore(args.repo_root)
        publisher = ResearchRecordPublisher.from_environment(store)
        if not publisher.sinks:
            raise ValueError("research publication replay requires a configured sink")
        result = publisher.replay(research_id=args.research_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["failed"]:
            return 1
    else:
        from fugue.research.skills import export_skill

        path = export_skill(args.destination)
        print(path)
    return 0


def _comparison_init(args: argparse.Namespace) -> int:
    from fugue.bench.comparison import scaffold_comparison

    path = scaffold_comparison(
        args.destination,
        template=args.template,
        force=args.force,
    )
    CONSOLE.print(f"[fugue.success]Created[/] {path}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    from fugue.doctor import doctor_report

    environment = load_env(args.env_file) if args.env_file is not None else None
    report = doctor_report(
        args.workspace,
        model=args.model,
        env=environment,
        required_capabilities=args.require,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        table = Table(title="Fugue doctor", box=box.ROUNDED)
        table.add_column("Capability")
        table.add_column("Status")
        table.add_column("Detail")
        distribution = report["distribution"]
        readiness = report["readiness"]
        assets_ready = readiness["requirements"]["packaged_assets"]["ready"]
        assets_detail = readiness["requirements"]["packaged_assets"]["detail"]
        table.add_row(
            "distribution",
            "ready" if assets_ready else "invalid",
            f"fugue {distribution['version']} · {distribution['digest'][:12]}",
        )
        table.add_row(
            "packaged assets",
            "ready" if assets_ready else "invalid",
            assets_detail,
        )
        if readiness["mode"] == "required":
            table.add_row(
                "requested readiness",
                "ready" if readiness["ready"] else "blocked",
                ", ".join(readiness["requested_capabilities"]),
            )
            for name, requirement in readiness["requirements"].items():
                table.add_row(
                    name.replace("_", " "),
                    "ready" if requirement["ready"] else "blocked",
                    requirement["detail"],
                )
        for name, item in report["optional_features"].items():
            ready = item.get("ready", item["installed"])
            detail = item.get("version") or "install the matching Fugue extra"
            table.add_row(
                name.replace("_", " "), "ready" if ready else "optional", detail
            )
        CONSOLE.print(table)
    return 0 if report["ok"] else 2


def _comparison_check(args: argparse.Namespace) -> int:
    from fugue.bench.comparison import check_comparison, load_comparison

    root, comparison = _comparison_cli_context(args)
    spec = load_comparison(comparison, repo_root=root)
    readiness = check_comparison(spec, repo_root=root)
    if args.json:
        print(json.dumps(readiness.to_dict(), indent=2, sort_keys=True))
    else:
        _print_comparison_readiness(
            readiness,
            evidence_mode=spec.execution.evidence_mode,
        )
    return 0 if readiness.status in {"ready", "needs_review"} else 2


def _comparison_prepare(
    args: argparse.Namespace,
    *,
    spec: Any,
    root: Path,
    operator: OperatorService,
) -> int:
    from fugue.bench.comparison import prepare_comparison

    receipt, preview, receipt_path = prepare_comparison(
        spec,
        repo_root=root,
        operator=operator,
    )
    preview_usable = (
        int(preview.matrix["applicable_cells"])
        == int(preview.readiness["estimated_cells"])
        == int(preview.matrix["estimated_trials"])
    )
    if args.json:
        payload = preview.to_dict()
        payload["approval_eligible"] = preview_usable
        payload["preparation"] = {
            "receipt": receipt,
            "path": receipt_path.relative_to(root).as_posix(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_comparison_readiness_dict(
            preview.readiness,
            evidence_mode=spec.execution.evidence_mode,
        )
        treatment_identities = _comparison_preview_treatment_identities(preview.matrix)
        prepared_summary = Text()
        prepared_summary.append(receipt["receipt_digest"], style="bold")
        prepared_summary.append(
            f"\n{receipt_path.relative_to(root)}\n"
            f"Final preview: {preview.preview_digest}"
        )
        _append_comparison_preview_treatment_identities(
            prepared_summary,
            treatment_identities,
        )
        CONSOLE.print(
            Panel(
                prepared_summary,
                title="Prepared comparison",
                border_style="fugue.cyan",
            )
        )
        if not preview_usable:
            CONSOLE.print(
                "[fugue.warning]Preparation completed, but the final "
                "preview is not approval-eligible in this environment.[/]"
            )
    return 0 if preview_usable else 2


def _comparison_compare(args: argparse.Namespace) -> int:
    from fugue.bench.comparison import (
        ComparisonPublicationError,
        check_comparison,
        execute_comparison,
        load_comparison,
        preview_comparison,
    )

    root, comparison = _comparison_cli_context(args)
    spec = load_comparison(comparison, repo_root=root)
    operator = OperatorService(root, args.env_file)
    if args.prepare:
        return _comparison_prepare(
            args,
            spec=spec,
            root=root,
            operator=operator,
        )
    readiness = check_comparison(spec, repo_root=root)
    if readiness.status not in {"ready", "needs_review"}:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema_version": 2,
                        "preview_digest": None,
                        "approval_eligible": False,
                        "readiness": readiness.to_dict(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            _print_comparison_readiness(
                readiness,
                evidence_mode=spec.execution.evidence_mode,
            )
            CONSOLE.print(
                "\n[fugue.warning]Exact preview not generated while the "
                "comparison is blocked.[/]"
            )
        return 2
    preview = preview_comparison(
        spec,
        repo_root=root,
        operator=operator,
    )
    preview_usable = (
        int(preview.matrix["applicable_cells"])
        == int(preview.readiness["estimated_cells"])
        == int(preview.matrix["estimated_trials"])
    )
    if args.preview:
        if args.json:
            payload = preview.to_dict()
            payload["approval_eligible"] = preview_usable
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_comparison_readiness_dict(
                preview.readiness,
                evidence_mode=spec.execution.evidence_mode,
            )
            if preview_usable:
                treatment_identities = _comparison_preview_treatment_identities(
                    preview.matrix
                )
                preview_summary = Text()
                preview_summary.append(preview.preview_digest, style="bold")
                preview_summary.append(
                    f"\n{preview.matrix['estimated_trials']} runnable aligned attempts\n"
                    "This exact digest is eligible for human approval."
                )
                _append_comparison_preview_treatment_identities(
                    preview_summary,
                    treatment_identities,
                )
                CONSOLE.print(
                    Panel(
                        preview_summary,
                        title="Exact preview",
                        border_style="fugue.cyan",
                    )
                )
            else:
                CONSOLE.print(
                    "\n[fugue.warning]No usable preview: one or more planned "
                    "attempts are not runnable in this environment.[/]"
                )
        return 0 if preview_usable else 2
    if not preview_usable:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema_version": 2,
                        "preview_digest": preview.preview_digest,
                        "approval_eligible": False,
                        "readiness": preview.readiness,
                        "matrix": preview.matrix,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            CONSOLE.print(
                "[fugue.warning]Comparison cannot run because one or more "
                "planned attempts are not runnable in this environment.[/]"
            )
        return 2
    if not args.approval:
        raise ValueError("--run or --resume requires --approval APPROVAL_DIGEST")
    publication_error: ComparisonPublicationError | None = None
    try:
        result, json_path, markdown_path = execute_comparison(
            preview,
            approval_digest=args.approval,
            repo_root=root,
            env_file=args.env_file,
            fetch_weave=args.fetch_weave,
            resume_run_id=args.resume,
        )
    except ComparisonPublicationError as exc:
        publication_error = exc
        if exc.result is None:
            payload = {
                "schema_version": 1,
                "status": "publication_incomplete",
                "stage": exc.stage,
                "research_id": exc.research_id,
                "error_type": exc.error_type,
                "receipt": exc.receipt_path.relative_to(root).as_posix(),
                "behavioral_result": None,
            }
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                CONSOLE.print(
                    "[fugue.warning]Research publication failed before any "
                    "comparison cells started.[/]"
                )
                CONSOLE.print(f"Publication receipt: {exc.receipt_path}")
            return 3
        if exc.result_path is None or exc.markdown_path is None:
            raise RuntimeError(
                "result publication failure did not preserve result paths"
            ) from exc
        result = exc.result
        json_path = exc.result_path
        markdown_path = exc.markdown_path
    if args.json:
        payload = result.to_dict()
        if publication_error is not None:
            payload = {
                "schema_version": 1,
                "status": "publication_incomplete",
                "stage": publication_error.stage,
                "research_id": publication_error.research_id,
                "error_type": publication_error.error_type,
                "receipt": publication_error.receipt_path.relative_to(root).as_posix(),
                "behavioral_result": payload,
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        CONSOLE.print(Markdown(markdown_path.read_text(encoding="utf-8")))
        CONSOLE.print(f"\nResult JSON: {json_path}")
        if publication_error is not None:
            CONSOLE.print(
                "\n[fugue.warning]The immutable behavioral result was saved, "
                "but its declared Research publication is incomplete.[/]"
            )
            CONSOLE.print(f"Publication receipt: {publication_error.receipt_path}")
    if publication_error is not None:
        return 3
    behavioral = getattr(result, "behavioral_summary", None)
    if (
        result.incomplete
        or result.required_evaluations_incomplete
        or getattr(behavioral, "status", "") in {"invalid", "incomplete"}
    ):
        return 3
    return (
        1
        if result.regressed
        or getattr(result, "mixed", 0)
        or getattr(behavioral, "status", "") in {"regressed", "mixed"}
        else 0
    )


def _comparison_preview_treatment_identities(
    matrix: Mapping[str, Any],
) -> tuple[tuple[str, str, str, str], ...]:
    """Return presentation rows from the preview's canonical matrix cells."""

    rows: set[tuple[str, str, str, str]] = set()
    matrix_cells = matrix.get("matrix_cells")
    if not isinstance(matrix_cells, list | tuple):
        return ()
    for raw_cell in matrix_cells:
        if not isinstance(raw_cell, Mapping):
            continue
        candidate_id = str(raw_cell.get("candidate_id") or "")
        if not candidate_id:
            continue
        variant_id = str(raw_cell.get("variant_id") or "")
        variant_label = str(
            raw_cell.get("variant_label") or variant_id or "Unlabelled treatment"
        )
        harness = str(raw_cell.get("harness") or "unknown harness")
        rows.add((variant_label, variant_id, harness, candidate_id))
    return tuple(sorted(rows, key=lambda row: (row[1], row[2], row[3], row[0])))


def _append_comparison_preview_treatment_identities(
    target: Text,
    identities: tuple[tuple[str, str, str, str], ...],
) -> None:
    if not identities:
        return
    target.append("\n\nTreatment identities", style="bold")
    for variant_label, variant_id, harness, candidate_id in identities:
        target.append(f"\n- {variant_label} ({variant_id}; {harness}): {candidate_id}")


def _comparison_approve(args: argparse.Namespace) -> int:
    from fugue.bench.comparison import prepared_candidate_definitions
    from fugue.research.approvals import ApprovalLedger
    from fugue.research.store import StudyStore

    if args.max_cells is not None and args.max_cells < 1:
        raise ValueError("--max-cells must be positive")
    store = StudyStore(args.repo_root.resolve())
    operation_id = args.operation_id or f"approve-{args.preview_digest[:20]}"
    candidate_definitions = prepared_candidate_definitions(
        args.preview_digest,
        repo_root=args.repo_root.resolve(),
    )
    approval = ApprovalLedger(store.path).approve(
        subject_kind="experiment",
        preview_digest=args.preview_digest,
        maximum_cost_usd=args.max_usd,
        maximum_cells=args.max_cells,
        approved_by=args.approved_by,
        operation_id=operation_id,
        expires_in_seconds=args.expires_in,
        candidate_definitions=candidate_definitions,
    )
    print(json.dumps(approval.to_dict(), indent=2, sort_keys=True))
    return 0


def _comparison_result(args: argparse.Namespace) -> int:
    from fugue.bench.comparison import (
        COMPARISON_RESULT_ROOT,
        ComparisonResultV3,
        _result_markdown,
        read_comparison_result,
    )
    from fugue.bench.local_evidence import LocalEvidenceStore
    from fugue.bench.local_publication import (
        verify_comparison_result_local_evidence,
    )

    root = args.repo_root.resolve()
    if args.append_invalidation is not None:
        from fugue.research.comparisons import append_comparison_invalidation

        receipt = append_comparison_invalidation(
            root,
            args.append_invalidation,
            research_id=args.research_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.research_id is not None:
        raise ValueError("--research-id requires --append-invalidation")
    result_path, _stored_markdown_path = _comparison_result_paths(
        root,
        args.comparison,
        result_root=root / COMPARISON_RESULT_ROOT,
    )
    result = read_comparison_result(result_path)
    verification_note = ""
    if isinstance(result, ComparisonResultV3):
        if result.local_evidence is not None:
            manifest_path = LocalEvidenceStore(root, result.source).manifest_path
            verify_comparison_result_local_evidence(result, manifest_path)
        elif (
            result.evidence_backend == "local"
            or result.local_chain_integrity == "reconciled"
        ):
            raise ValueError(
                "ComparisonResultV3 claims canonical local evidence without a "
                "bound local evidence manifest"
            )
        else:
            verification_note = (
                "\n## Verification boundary\n\n"
                "- This historical hosted V3 result passed canonical result "
                "validation but predates the bound local evidence ledger. Its "
                "hosted evidence is not requalified by `fugue result`.\n"
            )
    else:
        verification_note = (
            "\n## Verification boundary\n\n"
            "- This historical V1/V2 result passed its canonical result-digest "
            "validation but has no V3 local evidence manifest to reconcile. It "
            "remains readable audit history, not qualified local evidence.\n"
        )
    verified_markdown = _result_markdown(result) + verification_note
    reviewed_at = args.reviewed_at or datetime.now(UTC).isoformat().replace(
        "+00:00",
        "Z",
    )
    if args.authorize_followup is not None:
        from fugue.bench.comparison import authorize_comparison_followup

        if not args.reviewed_by:
            raise ValueError("--authorize-followup requires --reviewed-by")
        receipt, canonical_result, attestation = authorize_comparison_followup(
            result_path=result_path,
            followup_spec_path=args.authorize_followup,
            reviewed_by=args.reviewed_by,
            reviewed_at=reviewed_at,
            repo_root=root,
        )
        receipt["result_path"] = canonical_result.relative_to(root).as_posix()
        receipt["attestation_path"] = attestation.relative_to(root).as_posix()
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.signoff_by is not None:
        from fugue.bench.comparison import attest_comparison_decision

        signed = attest_comparison_decision(
            result_path=result_path,
            signer=args.signoff_by,
            signed_at=reviewed_at,
        )
        print(json.dumps(signed.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.reviewed_by or args.reviewed_at:
        raise ValueError(
            "--reviewed-by/--reviewed-at require --authorize-followup or --signoff-by"
        )
    if args.open_result:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="fugue-verified-result-",
            suffix=".md",
            delete=False,
        ) as handle:
            handle.write(verified_markdown)
            verified_markdown_path = Path(handle.name)
        verified_markdown_path.chmod(0o600)
        webbrowser.open(verified_markdown_path.resolve().as_uri())
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        CONSOLE.print(Markdown(verified_markdown))
    return 0


def _comparison_result_paths(
    root: Path,
    comparison: str,
    *,
    result_root: Path,
) -> tuple[Path, Path]:
    resolved_root = root.resolve()
    resolved_result_root = result_root.resolve()
    try:
        resolved_result_root.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("comparison result root escapes the repository") from exc
    if comparison == "latest":
        pointer_path = result_root / "latest.json"
        if not pointer_path.is_file() or pointer_path.is_symlink():
            raise FileNotFoundError("no comparison result has been recorded")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, Mapping):
            raise ValueError("latest comparison result pointer must be an object")
        result_path = _contained_result_pointer_path(
            root,
            resolved_result_root,
            pointer.get("result"),
            filename="result.json",
        )
        markdown_path = _contained_result_pointer_path(
            root,
            resolved_result_root,
            pointer.get("markdown"),
            filename="result.md",
        )
        if result_path.parent != markdown_path.parent:
            raise ValueError(
                "latest comparison result and Markdown pointers must share a directory"
            )
    else:
        result_path = None
        for candidate in sorted(result_root.glob("*/result.json")):
            contained_candidate = _contained_result_pointer_path(
                resolved_root,
                resolved_result_root,
                candidate,
                filename="result.json",
                allow_absolute=True,
            )
            candidate_payload = json.loads(
                contained_candidate.read_text(encoding="utf-8")
            )
            if (
                isinstance(candidate_payload, Mapping)
                and candidate_payload.get("comparison_id") == comparison
            ):
                result_path = contained_candidate
                break
        if result_path is None:
            raise FileNotFoundError(f"comparison result not found: {comparison}")
        markdown_path = _contained_result_pointer_path(
            resolved_root,
            resolved_result_root,
            result_path.with_name("result.md"),
            filename="result.md",
            allow_absolute=True,
        )
    if not result_path.is_file():
        raise FileNotFoundError(f"comparison result not found: {result_path}")
    return result_path, markdown_path


def _contained_result_pointer_path(
    root: Path,
    result_root: Path,
    raw: Any,
    *,
    filename: str,
    allow_absolute: bool = False,
) -> Path:
    if isinstance(raw, Path):
        selected = raw
    elif isinstance(raw, str) and raw.strip():
        selected = Path(raw)
    else:
        raise ValueError(f"comparison result pointer requires {filename}")
    if selected.is_absolute() and not allow_absolute:
        raise ValueError(f"comparison result pointer {filename} must be relative")
    resolved = (selected if selected.is_absolute() else root / selected).resolve()
    try:
        resolved.relative_to(result_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"comparison result pointer {filename} escapes the result root"
        ) from exc
    if resolved.name != filename:
        raise ValueError(f"comparison result pointer must name {filename}")
    return resolved


def _publish_local_result(args: argparse.Namespace) -> int:
    from fugue.bench.comparison import ComparisonResultV3, read_comparison_result
    from fugue.bench.local_evidence import LocalEvidenceStore
    from fugue.bench.local_publication import (
        LocalResultPublicationError,
        MissingWeaveExtraError,
        StudyPublicationScopeV1,
        publish_local_result_to_weave,
        weave_publication_target_from_environment,
        weave_publisher_from_environment,
    )
    from fugue.redaction import secrets_from_env

    root = args.repo_root.resolve()
    result_path = (
        args.result.resolve()
        if args.result.is_absolute()
        else (root / args.result).resolve()
    )
    result = read_comparison_result(result_path)
    if not isinstance(result, ComparisonResultV3):
        raise LocalResultPublicationError(
            "optional Weave publication requires ComparisonResultV3"
        )
    if args.study_id and not args.research_id:
        raise LocalResultPublicationError(
            "--study-id requires --research-id so collection membership is explicit"
        )
    study_scope = (
        StudyPublicationScopeV1(
            research_id=args.research_id,
            study_id=args.study_id or result.comparison_id,
        )
        if args.research_id
        else None
    )
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None and args.manifest.is_absolute()
        else (root / args.manifest).resolve()
        if args.manifest is not None
        else LocalEvidenceStore(root, result.source).manifest_path
    )
    receipt_path = (
        args.receipt.resolve()
        if args.receipt is not None and args.receipt.is_absolute()
        else (root / args.receipt).resolve()
        if args.receipt is not None
        else None
    )
    privacy_notice = (
        "Privacy boundary: publishing sends sanitized result and evidence-chain "
        "projections to the selected W&B project. Raw local transcript and "
        "tool-event artifact files remain local. Review the destination project's "
        "access policy before continuing."
    )
    if args.json:
        print(privacy_notice, file=sys.stderr)
    else:
        CONSOLE.print(
            privacy_notice.replace(
                "Privacy boundary:", "[fugue.gold]Privacy boundary:[/]"
            )
        )
    env = load_env(args.env_file)
    try:
        publisher = weave_publisher_from_environment(env)
        target = weave_publication_target_from_environment(
            args.project,
            env,
            study_scope=study_scope,
        )
        receipt = publish_local_result_to_weave(
            result_path,
            manifest_path,
            target=target,
            publisher=publisher,
            receipt_path=receipt_path,
            secret_values=secrets_from_env(env),
        )
    except MissingWeaveExtraError as exc:
        payload = {
            "schema_version": 1,
            "status": "blocked",
            "error_type": "missing_weave_extra",
            "message": str(exc),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            CONSOLE.print(f"[fugue.coral]Blocked:[/] {exc}")
        return 2
    if args.json:
        print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    else:
        scope_lines = (
            (
                f"Fugue Research scope: {receipt.target.study_scope.research_id}",
                f"Fugue Study key: {receipt.target.study_scope.study_id}",
            )
            if receipt.target.study_scope is not None
            else ()
        )
        CONSOLE.print(
            Panel(
                "\n".join(
                    (
                        f"Target: {receipt.target.project_slug}",
                        *scope_lines,
                        f"Result digest: {receipt.result_digest}",
                        f"Local evidence manifest digest: {receipt.local_manifest_digest}",
                        f"Hosted objects: {len(receipt.hosted_objects)}",
                        f"Publication receipt digest: {receipt.receipt_digest}",
                    )
                ),
                title="Published local result projection",
                border_style="fugue.success",
            )
        )
    return 0


def _study_index(args: argparse.Namespace) -> int:
    from fugue.bench.research_index import (
        ResearchIndexSourceV1,
        build_research_index,
        write_research_index,
    )
    from fugue.redaction import secrets_from_env

    root = args.repo_root.resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    sources = tuple(
        ResearchIndexSourceV1(
            result_path=resolve(result_path),
            publication_receipt_path=resolve(receipt_path),
        )
        for result_path, receipt_path in args.source
    )
    env = load_env(args.env_file)
    index = build_research_index(
        research_id=args.research_id,
        title=args.title,
        objective=args.objective,
        sources=sources,
        secret_values=secrets_from_env(env),
    )
    output = (
        args.output.resolve()
        if args.output is not None and args.output.is_absolute()
        else (root / args.output).resolve()
        if args.output is not None
        else (root / ".fugue" / "research" / args.research_id / "index.json")
    )
    write_research_index(
        output,
        index,
        secret_values=secrets_from_env(env),
    )
    if args.json:
        print(json.dumps(index.to_dict(), indent=2, sort_keys=True))
    else:
        CONSOLE.print(
            Panel(
                "\n".join(
                    (
                        f"Research: {index.research_id}",
                        f"Studies: {index.study_count}",
                        f"Result rows: {index.total_rows}",
                        f"Index digest: {index.index_digest}",
                        f"Path: {output}",
                    )
                ),
                title="Built immutable Research index",
                border_style="fugue.success",
            )
        )
    return 0


def _publish_research_index(args: argparse.Namespace) -> int:
    from fugue.bench.research_index import publish_research_index
    from fugue.bench.wandb_research_index import (
        MissingWandbIndexExtraError,
        wandb_research_index_publisher_from_environment,
        wandb_research_index_target_from_environment,
    )
    from fugue.redaction import secrets_from_env

    root = args.repo_root.resolve()
    index_path = (
        args.index.resolve()
        if args.index.is_absolute()
        else (root / args.index).resolve()
    )
    receipt_path = (
        args.receipt.resolve()
        if args.receipt is not None and args.receipt.is_absolute()
        else (root / args.receipt).resolve()
        if args.receipt is not None
        else None
    )
    env = load_env(args.env_file)
    try:
        publisher = wandb_research_index_publisher_from_environment(env)
        target = wandb_research_index_target_from_environment(args.project, env)
        receipt = publish_research_index(
            index_path,
            target=target,
            publisher=publisher,
            receipt_path=receipt_path,
            secret_values=secrets_from_env(env),
        )
    except MissingWandbIndexExtraError as exc:
        message = str(exc)
        payload = {
            "schema_version": 1,
            "status": "blocked",
            "error_type": "missing_wandb_index_extra",
            "message": message,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            CONSOLE.print(f"[fugue.coral]Blocked:[/] {message}")
        return 2
    if args.json:
        print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    else:
        report = receipt.report_url or "not created by fugue publish wandb-index"
        CONSOLE.print(
            Panel(
                "\n".join(
                    (
                        f"Project: {receipt.target.project}",
                        f"Destination: {receipt.target.destination_digest}",
                        f"Index Run: {receipt.run_url}",
                        f"Artifact version: {receipt.artifact_url}",
                        f"Report: {report}",
                        "Next action: run fugue publish wandb-report with this "
                        "index-publication receipt",
                        f"Index digest: {receipt.index_digest}",
                        f"Publication receipt digest: {receipt.receipt_digest}",
                    )
                ),
                title="Published W&B Research index",
                border_style="fugue.success",
            )
        )
    return 0


def _publish_research_report(args: argparse.Namespace) -> int:
    from fugue.bench.research_report import (
        ResearchIndexReportError,
        publish_research_index_report,
    )
    from fugue.bench.wandb_research_report import (
        MissingWandbReportExtraError,
        RetryableWandbResearchReportPublicationError,
        WandbResearchReportPublicationError,
        wandb_research_report_publisher_from_environment,
    )
    from fugue.redaction import secrets_from_env

    root = args.repo_root.resolve()
    index_path = (
        args.index.resolve()
        if args.index.is_absolute()
        else (root / args.index).resolve()
    )
    index_receipt_path = (
        args.index_receipt.resolve()
        if args.index_receipt.is_absolute()
        else (root / args.index_receipt).resolve()
    )
    receipt_path = (
        args.receipt.resolve()
        if args.receipt is not None and args.receipt.is_absolute()
        else (root / args.receipt).resolve()
        if args.receipt is not None
        else None
    )
    env = load_env(args.env_file)
    try:
        publisher = wandb_research_report_publisher_from_environment(env)
        receipt = publish_research_index_report(
            index_path,
            index_receipt_path,
            publisher,
            receipt_path=receipt_path,
            secret_values=secrets_from_env(env),
        )
    except MissingWandbReportExtraError as exc:
        payload = {
            "schema_version": 1,
            "status": "blocked",
            "error_type": "missing_wandb_report_extra",
            "message": str(exc),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            CONSOLE.print(f"[fugue.coral]Blocked:[/] {exc}")
        return 2
    except RetryableWandbResearchReportPublicationError as exc:
        payload = {
            "schema_version": 1,
            "status": "retryable",
            "error_type": "wandb_report_publication_retryable",
            "message": str(exc),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            CONSOLE.print(f"[fugue.coral]Retryable:[/] {exc}")
        return 3
    except (ResearchIndexReportError, WandbResearchReportPublicationError) as exc:
        payload = {
            "schema_version": 1,
            "status": "blocked",
            "error_type": "wandb_report_publication_blocked",
            "message": str(exc),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            CONSOLE.print(f"[fugue.coral]Blocked:[/] {exc}")
        return 2
    if args.json:
        print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    else:
        CONSOLE.print(
            Panel(
                "\n".join(
                    (
                        f"Project: {receipt.target.project}",
                        f"Report: {receipt.report_url}",
                        f"Projection digest: {receipt.projection_digest}",
                        "Verification: saved Report matches the projection digest",
                        f"Publication receipt digest: {receipt.receipt_digest}",
                        "Access: governed by W&B project and Report settings",
                    )
                ),
                title="W&B Report projection published (Reports API: Public Preview)",
                border_style="fugue.success",
            )
        )
    return 0


def _comparison_demo(args: argparse.Namespace) -> int:
    from fugue.bench.comparison import (
        analyze_comparison_rows,
        load_comparison,
        preview_comparison,
        score_comparison_rows,
        write_comparison_result,
    )
    from fugue.bench.export import write_jsonl

    root = args.repo_root.resolve()
    resource = files("fugue").joinpath("resources", "source-use-replay")
    with as_file(resource) as extracted:
        demo_root = Path(extracted)
        spec = load_comparison(
            demo_root / "comparison.yaml",
            repo_root=demo_root,
        )
        preview = preview_comparison(
            spec,
            repo_root=demo_root,
            operator=OperatorService(demo_root),
        )
        source_rows = [
            json.loads(line)
            for line in (demo_root / "attempts.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        rows = score_comparison_rows(
            spec,
            source_rows,
            repo_root=demo_root,
        )
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=rows,
        source="bundled-replay",
    )
    destination = (
        args.out.resolve() if args.out else root / "artifacts" / "source-use-replay"
    )
    write_jsonl(rows, destination / "attempts.jsonl")
    json_path, markdown_path = write_comparison_result(result, destination=destination)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        CONSOLE.print(
            Panel(
                "This is an immutable replay. It demonstrates Fugue's comparison "
                "and result workflow; it is not a new live experiment.",
                title="No-key replay",
                border_style="fugue.gold",
            )
        )
        CONSOLE.print(Markdown(markdown_path.read_text(encoding="utf-8")))
        CONSOLE.print(f"\nResult JSON: {json_path}")
    return 0 if not result.incomplete else 3


def _wandb_sandbox(args: argparse.Namespace) -> int:
    from fugue.bench.wandb_sandbox import (
        build_wandb_runtime,
        lock_wandb_runtime,
        read_wandb_runtime_lock,
        wandb_doctor,
    )

    if args.wandb_action == "build-runtime":
        root = args.repo_root.resolve()
        manifest = build_wandb_runtime(
            comparisons=args.comparisons,
            repo_root=root,
            platform=args.platform,
            image=args.image,
            push=args.push,
            output_manifest=args.output_manifest,
            sbom_dir=args.sbom_dir,
        )
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.wandb_action == "lock-runtime":
        lock = lock_wandb_runtime(args.manifest, output=args.output)
        print(json.dumps(lock.to_dict(), indent=2, sort_keys=True))
        return 0
    lock = read_wandb_runtime_lock(args.lock)
    result = wandb_doctor(lock, env_file=args.env_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["deleted"] and result["orphans"] == 0 else 3


def _print_comparison_readiness(
    readiness: Any,
    *,
    evidence_mode: str | None = None,
) -> None:
    _print_comparison_readiness_dict(
        readiness.to_dict(),
        evidence_mode=evidence_mode,
    )


def _print_comparison_readiness_dict(
    value: Mapping[str, Any],
    *,
    evidence_mode: str | None = None,
) -> None:
    status = str(value["status"])
    color = {
        "ready": "fugue.success",
        "needs_review": "fugue.gold",
        "blocked": "fugue.coral",
        "no_comparison_justified": "fugue.gold",
    }[status]
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Question", str(value["question"]))
    evidence_project = value.get("evidence_project")
    if evidence_mode == "local":
        evidence_destination = "canonical local artifact ledger"
    elif evidence_mode == "weave_required" and evidence_project:
        evidence_destination = (
            f"canonical local artifact ledger + required W&B/Weave: {evidence_project}"
        )
    elif evidence_mode == "weave_required":
        evidence_destination = (
            "canonical local artifact ledger + required hosted destination "
            "resolved by the operator"
        )
    else:
        evidence_destination = "inspect the exact preview evidence mode"
    table.add_row(
        "Evidence destination",
        evidence_destination,
    )
    table.add_row("Tasks", str(value["task_count"]))
    table.add_row("Changed", ", ".join(value["actual_changes"]) or "none")
    table.add_row(
        "Base check",
        f"{value['base_failures']}/{value['task_count']} intended failures",
    )
    table.add_row(
        "Gold check",
        f"{value['gold_passes']}/{value['task_count']} known-good passes",
    )
    judges = value.get("judge_evaluators") or []
    table.add_row("Judge", ", ".join(judges) if judges else "not used")
    table.add_row("Attempts", str(value["estimated_cells"]))
    table.add_row("Estimated cost", f"${float(value['estimated_cost_usd']):.2f}")
    table.add_row("Status", f"[{color}]{status}[/]")
    CONSOLE.print(table)
    for blocker in value.get("blockers") or []:
        CONSOLE.print(f"[fugue.coral]Blocked:[/] {blocker}")
    for warning in value.get("warnings") or []:
        CONSOLE.print(f"[fugue.gold]Review:[/] {warning}")


def _normalize_runs_argv(argv: list[str]) -> list[str]:
    """Keep the public `runs RUN_ID [ACTION]` grammar unambiguous to argparse."""
    if len(argv) >= 4 and argv[:2] == ["runs", "cancel"] and argv[2] == "--run-id":
        return ["runs", "--run-id", argv[3], "cancel", *argv[4:]]
    if (
        len(argv) >= 3
        and argv[:2] == ["runs", "cancel"]
        and not argv[2].startswith("-")
    ):
        return ["runs", "--run-id", argv[2], "cancel", *argv[3:]]
    if len(argv) < 2 or argv[0] != "runs" or argv[1].startswith("-"):
        return argv
    return ["runs", "--run-id", argv[1], *argv[2:]]


def _add_common_args(
    parser: argparse.ArgumentParser,
    *,
    json_output: bool = False,
    infer_comparison_root: bool = False,
) -> None:
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None if infer_comparison_root else Path(".env"),
        help="Read credentials from this file without copying it into the repository",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None if infer_comparison_root else Path.cwd(),
        help=argparse.SUPPRESS,
    )
    if json_output:
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit structured output without Rich decoration",
        )


def _comparison_cli_context(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve a comparison and its workspace without widening repository scope.

    An explicit ``--repo-root`` remains authoritative. When it is omitted, only
    a valid standalone-study marker beside the selected comparison permits the
    comparison directory to become the repository root. All other paths retain
    the historical current-working-directory behavior.
    """

    comparison = Path(args.comparison)
    explicit_root = getattr(args, "repo_root", None)
    if explicit_root is not None:
        return Path(explicit_root).resolve(), comparison

    resolved_comparison = comparison.resolve()
    inferred_root = _generated_study_root(resolved_comparison)
    if inferred_root is not None:
        return inferred_root, resolved_comparison
    return Path.cwd().resolve(), comparison


def _generated_study_root(comparison: Path) -> Path | None:
    marker = comparison.parent / _STANDALONE_STUDY_MARKER
    try:
        raw = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, Mapping):
        return None
    if (
        type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != 1
        or raw.get("kind") != "fugue_standalone_study"
        or raw.get("template") not in _STANDALONE_TEMPLATE_IDS
    ):
        return None
    return comparison.parent.resolve()


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, help="Benchmark manifest override")
    parser.add_argument("--harnesses", help="Comma-separated harness subset")
    parser.add_argument("--variants", help="Comma-separated variant subset")
    parser.add_argument("--preset", help="Saved experiment preset")
    parser.add_argument("--workloads", help="Comma-separated workload subset")
    parser.add_argument("--systems", help="Comma-separated context-system subset")
    parser.add_argument(
        "--model", help="Model selector: wandb/..., openai/..., anthropic/..."
    )
    parser.add_argument("--judge-model", help="Independent model route for QA judging")
    parser.add_argument(
        "--builder-model", help="Model route used to build generated context"
    )
    parser.add_argument("-k", "--n-attempts", type=_positive_cli_int)
    parser.add_argument("-n", "--n-concurrent", type=_positive_cli_int)
    parser.add_argument("-l", "--n-tasks", type=_positive_cli_int)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Read credentials from this file without copying it into the repository",
    )
    parser.add_argument("--jobs-dir", type=Path)
    parser.add_argument(
        "--cohort-id",
        help="Stable cohort identity recorded in the immutable run snapshot",
    )
    parser.add_argument(
        "--selection-lock",
        type=Path,
        help="Treatment selection lock required by preregistered confirmatory presets",
    )
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
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS
    )


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
    runs = service.runs(recover=False)
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
    readiness.add_row(
        "Credentials", _state(status.model_key_present and status.trace_key_present)
    )
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
                Panel(
                    readiness, title=f"Readiness {ready}/4", border_style="fugue.gold"
                ),
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
        glyph = (
            "■"
            if status == "running"
            else "▪"
            if status == "passed"
            else "×"
            if status == "failed"
            else "·"
        )
        lines.append(f"{label:<10} {glyph * 12}  {status.replace('_', ' ')}")
    return Panel("\n".join(lines), title="Harness sequencer", border_style="fugue.cyan")


def _print_preview(preview: Any) -> None:
    summary = Table.grid(padding=(0, 2))
    summary.add_row("Cells", str(preview.cells))
    summary.add_row("Applicable", str(preview.applicable_cells))
    summary.add_row("Runnable attempts", str(preview.estimated_trials))
    summary.add_row("Harnesses", ", ".join(preview.harnesses) or "none")
    summary.add_row("Variants", ", ".join(preview.variants) or "none")
    summary.add_row("Workloads", ", ".join(preview.workloads) or "none")
    summary.add_row(
        "Environment",
        str((preview.environment or {}).get("type") or "docker"),
    )
    evidence_destination = dict(preview.evidence_destination or {})
    if preview.evidence_project:
        evidence_summary = (
            "canonical local artifact ledger + required W&B/Weave: "
            f"{preview.evidence_project}"
        )
    elif evidence_destination.get("kind") == "local":
        evidence_summary = "canonical local artifact ledger"
    else:
        evidence_summary = (
            "canonical local artifact ledger + required hosted destination "
            "resolved by the operator"
        )
    summary.add_row("Evidence destination", evidence_summary)
    commands = "\n".join(preview.commands) or "No applicable commands."
    CONSOLE.print(
        Group(
            Panel(summary, title="Experiment matrix", border_style="fugue.gold"),
            Panel(
                Syntax(commands, "bash", word_wrap=True),
                title="Harbor commands",
                border_style="fugue.cyan",
            ),
        )
    )


def _print_draft(draft: Any) -> None:
    body = [draft.rationale or "No rationale supplied."]
    if draft.assumptions:
        body.append("\nAssumptions: " + "; ".join(draft.assumptions))
    if draft.assets:
        body.append(
            "\nAssets: " + ", ".join(f"{item.kind}:{item.id}" for item in draft.assets)
        )
    CONSOLE.print(
        Panel(
            "\n".join(body),
            title=f"{draft.experiment.title}  [{draft.experiment.id}]",
            border_style="fugue.gold",
        )
    )
    _print_preview(draft.preview)
    if draft.diff:
        CONSOLE.print(
            Panel(
                Syntax(draft.diff, "diff"),
                title="Proposed diff",
                border_style="fugue.cyan",
            )
        )
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
    if preview.selection is not None:
        table.add_row("Selection", preview.selection.decision.replace("_", " "))
        table.add_row("Candidate", preview.selection.selected_candidate_id or "none")
        table.add_row("Selection reason", preview.selection.reason)
    if scope.missing_metrics:
        table.add_row("Missing metrics", ", ".join(scope.missing_metrics))
    CONSOLE.print(
        Panel(table, title="Resolved analysis scope", border_style="fugue.cyan")
    )
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


def _print_service_statuses(statuses: Any) -> None:
    table = Table("Service", "State", "Detail", box=box.SIMPLE_HEAD)
    for status in statuses:
        table.add_row(
            status.service_id,
            _state(status.ready),
            f"{status.state}: {status.detail}",
        )
    if not statuses:
        table.add_row("none", "—", "no selected context system needs a service")
    CONSOLE.print(Panel(table, title="Managed services", border_style="fugue.gold"))


def _print_context_preparation(records: Any) -> None:
    table = Table(
        "System", "Variant / mode", "Task", "State", "Detail", box=box.SIMPLE_HEAD
    )
    for record in records:
        table.add_row(
            record.system_id,
            " / ".join(
                value for value in (record.variant_id, record.retrieval_mode) if value
            )
            or "—",
            record.task_id,
            record.status,
            record.detail,
        )
    if records:
        CONSOLE.print(
            Panel(table, title="Context preparation", border_style="fugue.cyan")
        )
    else:
        CONSOLE.print("[fugue.muted]No context artifacts were required.[/]")


def _run_command(args: argparse.Namespace) -> int:
    if args.run_id:
        return _run_worker(args)
    service = OperatorService(args.repo_root, args.env_file)
    try:
        experiment = _load_experiment_arg(args)
    except FileNotFoundError as exc:
        return _report_missing_run_asset(
            exc,
            repo_root=service.repo_root,
            as_json=args.json,
        )
    request = _request_from_args(args, experiment.id)
    inline_experiment = bool(
        args.experiment_file or args.manifest or not args.experiment
    )
    if args.preview:
        try:
            preview = (
                service.preview_experiment(experiment, request=request)
                if inline_experiment
                else service.preview(request)
            )
        except FileNotFoundError as exc:
            return _report_missing_run_asset(
                exc,
                repo_root=service.repo_root,
                as_json=args.json,
            )
        if args.json:
            from fugue.bench.operator import as_json

            print(as_json(preview))
        else:
            _print_preview(preview)
        return 0
    try:
        run = service.launch(
            request,
            experiment=experiment if inline_experiment else None,
        )
    except FileNotFoundError as exc:
        return _report_missing_run_asset(
            exc,
            repo_root=service.repo_root,
            as_json=args.json,
        )
    if args.json:
        from fugue.bench.operator import as_json

        final = run if args.detach else service.wait_for_run(run.run_id)
        print(as_json(final))
        return 0 if final.status in {"starting", "running", "passed"} else 1
    if args.detach:
        _print_started_run(run)
        return 0
    return _wait_for_run(service, run.run_id)


def _report_missing_run_asset(
    exc: FileNotFoundError,
    *,
    repo_root: Path,
    as_json: bool,
) -> int:
    raw_path = Path(exc.filename) if exc.filename else None
    if raw_path is not None:
        try:
            display_path = (
                raw_path.resolve().relative_to(repo_root.resolve()).as_posix()
            )
        except ValueError:
            display_path = raw_path.name
    else:
        display_path = "unknown"
    payload = {
        "schema_version": 1,
        "status": "blocked",
        "error_type": "missing_governed_asset",
        "asset": display_path,
        "next_action": (
            "Materialize and lock the referenced task or evaluation asset before "
            "previewing or running this experiment."
        ),
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        CONSOLE.print(
            Panel(
                f"[bold]{display_path}[/]\n{payload['next_action']}",
                title="Experiment blocked: missing governed asset",
                border_style="fugue.warning",
            )
        )
    return 2


def _run_worker(args: argparse.Namespace) -> int:
    run_id = getattr(args, "run_id", None) or new_run_id()
    experiment = _load_experiment_arg(args)
    service = OperatorService(args.repo_root, args.env_file)
    request = _request_from_args(args, experiment.id)
    final = service.execute_run(
        request,
        run_id=run_id,
        experiment=experiment,
    )
    if getattr(args, "json", False):
        from fugue.bench.operator import as_json

        print(as_json(final))
    else:
        CONSOLE.print(
            f"[bold]run {run_id}[/]: {final.passed} passed, "
            f"{final.failed} failed, {final.not_applicable} not applicable"
        )
    return 0 if final.status == "passed" else 1


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
        cohort_id=getattr(args, "cohort_id", None),
        selection_lock=getattr(args, "selection_lock", None),
    )


def _tui(args: argparse.Namespace) -> int:
    from fugue.tui import run_tui

    screen = "compose" if args.screen == "plan" else args.screen
    run_tui(initial_screen=screen, experiment_id=args.experiment)
    return 0


def _component_mcp(args: argparse.Namespace) -> int:
    if args.mcp_action == "prepare-wandb-release":
        from fugue.reference_studies.wandb_mcp import (
            HUMAN_READABLE_COMPARISON_NAME,
            prepare_wandb_mcp_reference_study,
        )

        receipt = prepare_wandb_mcp_reference_study(
            repo_root=args.repo_root.resolve(),
            env_file=args.env_file.resolve() if args.env_file is not None else None,
            platform=args.platform,
        )
        payload = receipt.to_dict()
        materialization = payload.get("materialization")
        artifact_paths = {
            str(item.get("path") or "")
            for item in (
                materialization.get("artifacts")
                if isinstance(materialization, Mapping)
                else ()
            )
            if isinstance(item, Mapping)
        }
        if "comparison.yaml" not in artifact_paths:
            raise RuntimeError(
                "W&B MCP preparation did not materialize a runnable comparison"
            )
        if HUMAN_READABLE_COMPARISON_NAME not in artifact_paths:
            raise RuntimeError(
                "W&B MCP preparation did not materialize the human-readable "
                "evidence canary"
            )
        comparison_path = (
            args.repo_root.resolve() / str(payload["destination"]) / "comparison.yaml"
        )
        payload.update(
            candidate_sha=str(payload["source_commit"]),
            comparison_path=comparison_path.as_posix(),
            human_readable_comparison_path=(
                args.repo_root.resolve()
                / str(payload["destination"])
                / HUMAN_READABLE_COMPARISON_NAME
            ).as_posix(),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    from fugue.bench.component_imports import (
        add_mcp_command,
        import_mcp_config,
        inspect_mcp_import,
        lock_mcp_import,
    )

    root = args.repo_root.resolve()
    if args.mcp_action == "import":
        value: Any = import_mcp_config(
            args.config,
            server=args.server,
            import_id=args.import_id,
            repo_root=root,
            allowed_hosts=tuple(args.allow_host),
        )
        payload = value.to_dict()
    elif args.mcp_action == "add":
        argv = list(args.argv)
        if argv[:1] == ["--"]:
            argv = argv[1:]
        value = add_mcp_command(
            args.import_id,
            argv,
            repo_root=root,
            required_env=tuple(args.required_env),
            allowed_hosts=tuple(args.allow_host),
        )
        payload = value.to_dict()
    elif args.mcp_action == "inspect":
        payload = inspect_mcp_import(args.import_id, root)
    else:
        value = lock_mcp_import(
            args.import_id,
            root,
            acknowledge_package_code=args.acknowledge_package_code,
            target_platform=args.platform,
        )
        payload = value.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _component_taskset(args: argparse.Namespace) -> int:
    from fugue.bench.tasksets import (
        import_weave_dataset,
        write_taskset_schemas,
    )

    if args.taskset_action == "schema":
        public, private = write_taskset_schemas(args.destination.resolve())
        payload: Any = {
            "public_task_schema": public.as_posix(),
            "private_label_schema": private.as_posix(),
        }
    else:
        root = args.repo_root.resolve()
        payload = import_weave_dataset(
            args.dataset,
            import_id=args.import_id,
            repo_root=root,
            env=load_env(args.env_file),
        ).to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _component_provider(args: argparse.Namespace) -> int:
    from fugue.bench.providers import (
        lock_provider,
        provider_conformance,
        scaffold_provider,
        validate_provider,
        write_provider_schemas,
    )

    if args.provider_action == "validate":
        payload: Any = validate_provider(args.command, timeout_sec=args.timeout)
    elif args.provider_action == "lock":
        payload = lock_provider(
            args.command,
            output=args.output,
            timeout_sec=args.timeout,
        ).to_dict()
    elif args.provider_action == "conformance":
        payload = provider_conformance(
            provider_lock=args.provider,
            candidate_ref=args.candidate,
            suite_ref=args.suite,
            exercise_run_cell=args.exercise_run_cell,
            timeout_sec=args.timeout,
            task_ids=args.tasks,
        )
        if args.output is not None:
            from fugue.bench.files import atomic_write_json

            atomic_write_json(args.output.resolve(), payload, mode=0o600)
    elif args.provider_action == "scaffold":
        provider_path, readme_path = scaffold_provider(
            args.destination,
            provider_id=args.provider_id,
            force=args.force,
        )
        payload = {
            "provider": provider_path.as_posix(),
            "readme": readme_path.as_posix(),
        }
    else:
        paths = write_provider_schemas(args.destination)
        payload = {"schemas": [path.as_posix() for path in paths]}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _component_skills(args: argparse.Namespace) -> int:
    from fugue.bench.component_imports import (
        import_skill,
        inspect_skill_import,
        lock_skill_import,
    )

    root = args.repo_root.resolve()
    if args.skills_action == "import":
        value: Any = import_skill(
            args.source,
            repo_root=root,
            import_id=args.import_id,
        )
        payload = value.to_dict()
    elif args.skills_action == "inspect":
        payload = inspect_skill_import(args.skill_id, root)
    else:
        value = lock_skill_import(args.skill_id, root)
        payload = value.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
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
            save_id = Prompt.ask(
                "Experiment id", default=f"{draft.experiment.id}-planned"
            )
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
        if (
            draft.experiment.trace_content == "full"
            and CONSOLE.is_terminal
            and not args.yes
        ):
            if not Confirm.ask(
                "Run with full prompt, response, and tool content in Weave?"
            ):
                run_requested = False
        if not run_requested:
            return 0
        saved = (
            composer.save(
                draft, experiment_id=save_id, replace_assets=args.replace_assets
            )
            if save_id and saved is None
            else saved
        )
        if draft.assets and not saved:
            raise ValueError(
                "save the experiment and all proposed assets before running; "
                f"rerun `fugue plan {' '.join(args.request)}` with --save"
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
            raise ValueError(
                "analysis question is required unless --saved or --list is used"
            )
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
            CONSOLE.print(
                "[fugue.muted]Scope only. Use --yes to generate the report.[/]"
            )
        return 0
    result = asyncio.run(analyst.execute(preview, model=args.model))
    if args.json:
        print(as_json(result))
    else:
        CONSOLE.print(
            Panel(Markdown(result.report), title="Analysis", border_style="fugue.cyan")
        )
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
        harnesses=tuple(_csv(args.harnesses) or []),
        variants=tuple(_csv(args.variants) or []),
        model=args.model,
        builder_model=args.builder_model,
        judge_model=args.judge_model,
        n_attempts=args.n_attempts,
        n_tasks=args.n_tasks,
        n_concurrent=args.n_concurrent,
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
    service_result = _setup_service_action(args, service, request, as_json)
    if service_result is not None:
        return service_result
    if args.prepare:
        prepared = service.prepare(request, rebuild=args.rebuild)
        if args.json:
            print(as_json(prepared))
        else:
            _print_context_preparation(prepared.context)
            for runtime in prepared.agent_runtimes:
                CONSOLE.print(
                    f"[fugue.success]{runtime.harness}[/] {runtime.status}: "
                    f"{runtime.image} [{runtime.architecture}] "
                    f"({runtime.image_id[:19]})"
                )
            for runtime in prepared.task_runtimes:
                verification = (
                    " base-fail/gold-pass verified"
                    if runtime.verification
                    and runtime.verification.get("base_failed") is True
                    and runtime.verification.get("gold_passed") is True
                    else ""
                )
                CONSOLE.print(
                    f"[fugue.success]task {runtime.task_id}[/] {runtime.status}: "
                    f"{runtime.image} [{runtime.architecture}] "
                    f"({runtime.image_id[:19]}){verification}"
                )
            for dataset in prepared.workload_datasets:
                CONSOLE.print(
                    f"[fugue.success]dataset {dataset.dataset_id}[/] "
                    f"{dataset.status}: {dataset.sample_count} samples "
                    f"({dataset.sha256[:19]})"
                )
            if prepared.portable_context_runtime is not None:
                runtime = prepared.portable_context_runtime
                CONSOLE.print(
                    f"[fugue.success]{runtime.harness}[/] {runtime.status}: "
                    f"{runtime.image} [{runtime.architecture}] "
                    f"({runtime.image_id[:19]})"
                )
        return 0 if all(item.status != "skipped" for item in prepared.context) else 1
    if args.prepare_context:
        records = service.prepare_context(request, rebuild=args.rebuild)
        if args.json:
            print(as_json(records))
        else:
            _print_context_preparation(records)
        return 0 if all(item.status != "skipped" for item in records) else 1
    if args.skills:
        inspections = service.prepare_skills(request, refresh=args.refresh_skills)
        if args.json:
            print(as_json(inspections))
        else:
            CONSOLE.print_json(as_json(inspections))
            CONSOLE.print(
                "[fugue.muted]Review the inventory and findings, then approve with "
                "--approve-skill ID=sha256:…[/]"
            )
        return 0
    if args.approve_skill:
        skill_id, separator, digest = args.approve_skill.partition("=")
        if not separator or not skill_id or not digest:
            raise ValueError("--approve-skill must use ID=DIGEST")
        entry = service.approve_skill(
            skill_id,
            digest,
            acknowledged_findings=tuple(args.acknowledge_risk),
        )
        if args.json:
            print(as_json(entry))
        else:
            CONSOLE.print_json(as_json(entry))
        return 0
    status = service.status(request)
    if args.json:
        print(as_json(status))
    else:
        _print_setup(status)
    return (
        0
        if all(route.key_present for route in status.routes)
        and status.trace_key_present
        else 1
    )


def _setup_service_action(args, service, request, as_json) -> int | None:
    actions = (
        (args.start_services, service.start_services, True),
        (args.service_status, service.service_status, True),
        (args.stop_services, service.stop_services, False),
    )
    selected = next((item for item in actions if item[0]), None)
    if selected is None:
        return None
    _, action, expected_ready = selected
    statuses = action(request)
    if args.json:
        print(as_json(statuses))
    else:
        _print_service_statuses(statuses)
    return 0 if all(item.ready is expected_ready for item in statuses) else 1


def _runs(args: argparse.Namespace) -> int:
    from fugue.bench.operator import OperatorService, as_json

    service = OperatorService(args.repo_root, args.env_file)
    if not args.run_id:
        if args.runs_action:
            raise ValueError("a run id is required for this action")
        return _runs_list(args, service, as_json)
    handlers = {
        "logs": _runs_logs,
        "cancel": _runs_cancel,
        "package": _runs_package,
        "export": _runs_export,
        "open": _runs_open,
    }
    if handler := handlers.get(args.runs_action):
        return handler(args, service, as_json)
    run = service.run_summary(args.run_id, recover=False)
    if args.json:
        print(as_json(run))
    else:
        CONSOLE.print(_run_panel(run))
        CONSOLE.print(_candidates_table(run))
        CONSOLE.print(_cells_table(run))
    return 0


def _runs_list(args: argparse.Namespace, service: Any, as_json: Any) -> int:
    runs = service.runs(recover=False)[: args.limit]
    if args.json:
        print(as_json(runs))
        return 0
    table = Table(title="Recent runs", box=box.SIMPLE_HEAD)
    for name in (
        "Run",
        "Experiment",
        "Status",
        "Passed",
        "Failed",
        "Cancelled",
        "Interrupted",
        "Pending",
    ):
        table.add_column(name)
    for run in runs:
        table.add_row(
            run.run_id,
            run.experiment_id,
            _status_markup(run.status),
            str(run.passed),
            str(run.failed),
            str(run.cancelled),
            str(run.interrupted),
            str(run.pending),
        )
    if runs:
        CONSOLE.print(table)
    else:
        CONSOLE.print("[fugue.muted]No runs yet. Start one with `fugue run pilot`.[/]")
    return 0


def _runs_logs(args: argparse.Namespace, service: Any, _: Any) -> int:
    if not args.follow:
        print(
            service.supervisor.read_log(
                args.run_id,
                cell_id=args.cell,
                recover=False,
            ),
            end="",
        )
        return 0
    try:
        for chunk in service.supervisor.follow_log(
            args.run_id,
            cell_id=args.cell,
            recover=False,
        ):
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        return 130
    return 0


def _runs_cancel(args: argparse.Namespace, service: Any, as_json: Any) -> int:
    run = service.supervisor.cancel(args.run_id)
    if args.json:
        print(as_json(service.run_summary(args.run_id)))
    else:
        CONSOLE.print(f"{run.run_id}: {_status_markup(run.status)}")
    return 0


def _runs_package(args: argparse.Namespace, service: Any, as_json: Any) -> int:
    if not args.yes:
        if not CONSOLE.is_terminal:
            raise ValueError("use --yes to confirm packaging non-interactively")
        confirmed = Confirm.ask(
            f"Package candidate {args.candidate} from run {args.run_id} "
            f"(allow failed: {'yes' if args.allow_failed else 'no'}) as {args.image}?"
        )
        if not confirmed:
            return 1
    result = service.package_candidate(
        args.run_id,
        args.candidate,
        workspace=args.workspace,
        image=args.image,
        platform=args.platform,
        allow_failed=args.allow_failed,
    )
    if args.json:
        print(as_json(result))
    else:
        CONSOLE.print(
            f"Packaged [bold]{result.candidate_id}[/] as "
            f"[cyan]{result.image}[/] ({result.deployment_id})"
        )
        CONSOLE.print(f"Deployment: [cyan]{result.path}[/]")
    return 0


def _runs_export(args: argparse.Namespace, service: Any, as_json: Any) -> int:
    summary = service.export_run(
        args.run_id,
        out=args.out,
        fetch_weave=args.fetch_weave,
        to_weave=args.to_weave,
        republish=args.republish,
        republish_reason=args.republish_reason,
    )
    if args.json:
        print(as_json(summary))
        return 0
    if summary.published:
        CONSOLE.print(
            f"Published {summary.published} finalized candidate evaluation(s)"
        )
        for evaluation in summary.evaluations:
            suffix = f" [cyan]{evaluation.url}[/]" if evaluation.url else ""
            CONSOLE.print(
                f"  {evaluation.name} ({evaluation.examples} examples; "
                f"{evaluation.linked_agent_predictions}/"
                f"{evaluation.agent_predictions} Agent-linked; "
                f"{evaluation.direct_predictions} direct){suffix}"
            )
            for reason in evaluation.linking_failures:
                CONSOLE.print(f"    [red]{reason}[/]")
    if summary.skipped:
        CONSOLE.print(f"Skipped {summary.skipped} published candidate(s)")
    for failure in summary.publication_failures:
        CONSOLE.print(f"[red]Publication failed:[/] {failure}")
    CONSOLE.print(f"Exported {summary.rows} rows to [cyan]{summary.path}[/]")
    return 0


def _runs_open(args: argparse.Namespace, service: Any, as_json: Any) -> int:
    links = service.run_links(args.run_id)
    url = links.project if args.destination == "project" else links.agents
    conversation_id = None
    if args.destination == "evaluation":
        evaluation = service.run_evaluation(args.run_id, cell_id=args.cell)
        if evaluation is None or evaluation.url is None:
            raise ValueError("run has no linked Weave evaluation")
        url = evaluation.url
    if args.destination == "trace":
        url = links.trace or links.agents
        refs = service.run_trace_refs(args.run_id, cell_id=args.cell)
        conversation_id = next(
            (value for reference in refs for value in reference.conversation_ids),
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


def _print_started_run(run: Any) -> None:
    CONSOLE.print(
        Panel(
            f"[fugue.success]started[/] [bold]{run.run_id}[/]\n"
            f"Logs: [fugue.cyan]{run.log_path}[/]\n"
            f"Follow: [bold]fugue runs {run.run_id} logs --follow[/]",
            title=run.run_name,
            border_style="fugue.success",
        )
    )


def _run_panel(run: Any) -> Panel:
    details = (
        f"{_status_markup(run.status)}  "
        f"[fugue.success]{run.passed} passed[/]  "
        f"[fugue.coral]{run.failed} failed[/]  "
        f"[yellow]{run.cancelled} cancelled[/]  "
        f"[yellow]{run.interrupted} interrupted[/]  "
        f"{run.pending} pending  {run.not_applicable} not applicable"
    )
    if run.cancellation_cleanup_status:
        details += (
            "\nCancellation cleanup: "
            f"{run.cancellation_cleanup_status} "
            f"({len(run.cancellation_cleanup_projects)} Compose projects)"
        )
        for error in run.cancellation_cleanup_errors:
            details += f"\n  [fugue.coral]{error}[/]"
    if run.evaluations:
        details += "\n\nWeave evaluations:"
        for evaluation in run.evaluations:
            if evaluation.url:
                details += (
                    f"\n  [link={evaluation.url}]{evaluation.name}[/link] "
                    f"({evaluation.linked_agent_predictions}/"
                    f"{evaluation.agent_predictions} Agent-linked; "
                    f"{evaluation.direct_predictions} direct)"
                )
            for reason in evaluation.linking_failures:
                details += f"\n    [fugue.coral]{reason}[/]"
    for failure in run.evaluation_failures:
        details += f"\n[fugue.coral]Observability:[/] {failure}"
    return Panel(
        details,
        title=f"{run.run_name}  [{run.run_id}]",
        border_style="fugue.cyan"
        if run.status in {"starting", "running"}
        else "fugue.gold",
    )


def _cells_table(run: Any) -> Table:
    table = Table(
        "Harness",
        "Variant",
        "Context",
        "Transport",
        "Task",
        "Candidate",
        "Execution",
        "Outcome",
        "Reason",
        box=box.SIMPLE_HEAD,
    )
    for cell in run.cells:
        table.add_row(
            cell.harness,
            cell.variant_id,
            cell.context_system_id,
            cell.context_delivery,
            cell.task_id,
            cell.candidate_id,
            _status_markup(cell.status),
            cell.benchmark_outcome.replace("_", " "),
            cell.error or cell.skip_reason or "-",
        )
    if not run.cells:
        table.add_row(
            "-",
            "-",
            "-",
            "-",
            "waiting for planner",
            "-",
            _status_markup(run.status),
            "-",
            "-",
        )
    return table


def _candidates_table(run: Any) -> Table:
    table = Table(
        "Candidate",
        "Configuration",
        "Passed",
        "Eval failed",
        "Exec failed",
        "Cancelled",
        "Interrupted",
        "Unscored",
        "Pending",
        "N/A",
        "Packageability",
        box=box.SIMPLE_HEAD,
    )
    for candidate in run.candidates:
        configuration = candidate.configuration
        table.add_row(
            candidate.display_id,
            " / ".join(
                str(value)
                for value in (
                    configuration.get("harness"),
                    configuration.get("model"),
                    (configuration.get("context") or {}).get("id"),
                )
                if value
            ),
            str(candidate.passed),
            str(candidate.failed),
            str(candidate.execution_failed),
            str(candidate.cancelled),
            str(candidate.interrupted),
            str(candidate.unscored),
            str(candidate.pending),
            str(candidate.not_applicable),
            candidate.packageability_reason,
        )
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
                        Panel(
                            log_tail or "Waiting for output...",
                            title="Live log",
                            border_style="fugue.muted",
                        ),
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
    variant = next(
        (item for item in experiment.variants if item.id == args.variant), None
    )
    if variant is None or variant.context.system_id != args.system:
        raise ValueError(
            f"variant {args.variant!r} does not select context system {args.system!r}"
        )
    runtime_env = load_env(args.env_file)
    runtime_env["FUGUE_CONTEXT_DELIVERY"] = variant.context.delivery
    runtime_env["FUGUE_VARIANT_ID"] = variant.id
    runtime = ContextRuntime(
        repo_root=args.repo_root,
        cache_root=args.repo_root / DEFAULT_CACHE_ROOT,
        env=runtime_env,
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
            context_config=variant.context.config,
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
