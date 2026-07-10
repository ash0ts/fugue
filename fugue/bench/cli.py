from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from fugue.bench.export import (
    export_rows,
    publish_to_weave,
    write_jsonl,
    write_parquet,
)
from fugue.bench.job_config import render_jobs
from fugue.bench.library import (
    ExperimentSpec,
    FeatureVariant,
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
from fugue.bench.manifest import BenchmarkManifest, load_manifest, write_lock
from fugue.bench.memory import (
    build_artifact,
    clone_repo_at_commit,
    write_memory_instruction,
)
from fugue.bridge import bridge_status, bridge_up, write_bridge_files
from fugue.model_plane import resolve_model_route, select_model, trace_env_defaults
from fugue.preflight import print_preflight, run_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fugue")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build memory artifacts")
    _add_experiment_arg(prepare)
    prepare.add_argument("--memory-variants", help="Comma-separated memory variants")
    prepare.add_argument("--repo-root", type=Path, default=Path.cwd())

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
    export.add_argument("--weave-project")

    preflight = subparsers.add_parser("preflight", help="Validate Fugue setup")
    preflight.add_argument("--model")
    preflight.add_argument("--env-file", type=Path, default=Path(".env"))
    preflight.add_argument("--repo-root", type=Path, default=Path.cwd())
    preflight.add_argument("--no-live", action="store_true")
    preflight.add_argument("--no-bridge-up", action="store_true")

    bridge = subparsers.add_parser("bridge", help="Manage the LiteLLM bridge")
    bridge_subparsers = bridge.add_subparsers(dest="bridge_command", required=True)
    bridge_up_parser = bridge_subparsers.add_parser("up", help="Render config and start bridge")
    bridge_up_parser.add_argument("--model")
    bridge_up_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    bridge_up_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    bridge_render = bridge_subparsers.add_parser("render", help="Render bridge files only")
    bridge_render.add_argument("--model")
    bridge_render.add_argument("--env-file", type=Path, default=Path(".env"))
    bridge_render.add_argument("--repo-root", type=Path, default=Path.cwd())
    bridge_subparsers.add_parser("status", help="Check bridge health")

    web = subparsers.add_parser("web", help="Run local Fugue operator UI")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)

    _add_library_commands(subparsers)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        return _prepare(args)
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
    raise AssertionError(args.command)


def _add_experiment_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", help="Saved experiment id")
    parser.add_argument("--manifest", type=Path, help="Benchmark manifest override")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    _add_experiment_arg(parser)
    parser.add_argument("--harnesses", help="Comma-separated harness subset")
    parser.add_argument("--variants", help="Comma-separated variant subset")
    parser.add_argument("--model", help="Model selector: wandb/..., openai/..., anthropic/...")
    parser.add_argument("-k", "--n-attempts", type=int)
    parser.add_argument("-n", "--n-concurrent", type=int)
    parser.add_argument("-l", "--n-tasks", type=int)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--jobs-dir", type=Path)
    parser.add_argument(
        "--run-name",
        help="W&B/Weave run grouping name. Defaults to FUGUE_RUN_NAME or a timestamp.",
    )
    parser.add_argument("--tags", help="Comma-separated extra W&B/Weave tags")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())


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


def _prepare(args: argparse.Namespace) -> int:
    experiment = _load_experiment_arg(args)
    manifest_path = _manifest_path_from_args(args, experiment)
    manifest = load_manifest(manifest_path)
    memory_variants = manifest.select_memory_variants(
        _csv(args.memory_variants) or _experiment_memory_variants(experiment)
    )
    artifact_root = _resolve(args.repo_root, manifest.artifact_root)
    checkout_root = artifact_root / "_checkouts"
    lock_records = []

    for memory in memory_variants:
        write_memory_instruction(artifact_root, memory)

    for task in manifest.tasks:
        checkout: Path | None = None
        for memory in memory_variants:
            if memory == "none":
                artifact = build_artifact(
                    memory=memory,
                    task=task,
                    repo_checkout=Path("."),
                    artifact_root=artifact_root,
                )
            else:
                if checkout is None:
                    checkout = clone_repo_at_commit(task, checkout_root)
                artifact = build_artifact(
                    memory=memory,
                    task=task,
                    repo_checkout=checkout,
                    artifact_root=artifact_root,
                )
            lock_records.append(artifact.to_lock_record(args.repo_root))

    write_lock(
        path=_resolve(args.repo_root, manifest.lock_path),
        manifest_path=manifest_path,
        manifest=manifest,
        artifacts=lock_records,
    )
    print(f"prepared {len(lock_records)} artifact records")
    return 0


def _run(args: argparse.Namespace) -> int:
    rendered = _rendered_jobs_from_args(args)
    for job in rendered:
        print("+ " + " ".join(shlex.quote(part) for part in job.command))
        print(f"# config: {job.config_path}")
        if not args.dry_run:
            subprocess.run(job.command, check=True, env=job.env)
    return 0


def _render(args: argparse.Namespace) -> int:
    rendered = _rendered_jobs_from_args(args)
    for job in rendered:
        print("+ " + " ".join(shlex.quote(part) for part in job.command))
        print(f"# config: {job.config_path}")
    print(f"rendered {len(rendered)} Harbor job config(s)")
    return 0


def _rendered_jobs_from_args(args: argparse.Namespace):
    experiment = _load_experiment_arg(args)
    experiment = _experiment_with_cli_overrides(experiment, args)
    manifest_path = _manifest_path_from_args(args, experiment)
    manifest = load_manifest(manifest_path)
    env = _load_env(args.env_file)
    env |= trace_env_defaults(env)
    run_name = _run_name(args.run_name, env)
    env["FUGUE_RUN_NAME"] = run_name
    env["FUGUE_RUN_GROUP"] = env.get("FUGUE_RUN_GROUP", "").strip() or run_name
    env["FUGUE_TAGS"] = ",".join(
        _run_tags(
            env=env,
            cli_tags=args.tags,
            run_name=run_name,
            manifest=manifest,
            manifest_path=manifest_path,
        )
    )
    return render_jobs(
        experiment=experiment,
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=args.repo_root,
        env=env,
        model=args.model,
        harness_names=_csv(args.harnesses),
        n_tasks=args.n_tasks,
        n_attempts=args.n_attempts,
        n_concurrent=args.n_concurrent,
        jobs_dir=args.jobs_dir,
        run_name=run_name,
        tags=_csv(args.tags) or [],
        run_id=_slug(run_name),
    )


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
        variants=[FeatureVariant(id="baseline", label="Baseline", memory="none")],
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


def _experiment_memory_variants(experiment: ExperimentSpec) -> list[str]:
    values = []
    for variant in experiment.variants:
        memory = variant.memory or "none"
        if variant.enabled and memory not in values:
            values.append(memory)
    return values or ["none"]


def _export(args: argparse.Namespace) -> int:
    rows = export_rows(
        args.jobs,
        fetch_weave=args.fetch_weave,
        weave_project=args.weave_project,
    )
    write_jsonl(rows, args.out)
    parquet_out = args.parquet_out or args.out.with_suffix(".parquet")
    if not write_parquet(rows, parquet_out):
        print("pyarrow is not installed; skipped parquet export")
    if args.to_weave:
        publish_to_weave(rows, args.weave_project)
    print(f"exported {len(rows)} rows to {args.out}")
    return 0


def _preflight(args: argparse.Namespace) -> int:
    env = _load_env(args.env_file)
    checks = run_preflight(
        args.model,
        repo_root=args.repo_root,
        env=env,
        live=not args.no_live,
        start_bridge=not args.no_bridge_up,
    )
    return print_preflight(checks)


def _bridge(args: argparse.Namespace) -> int:
    if args.bridge_command == "status":
        status = bridge_status()
        print(status)
        return 0 if status.get("ok") else 1

    env = _load_env(args.env_file)
    selected_model = select_model(args.model, env=env)
    route = resolve_model_route(selected_model, env)
    if args.bridge_command == "render":
        files = write_bridge_files(route, args.repo_root)
        print(f"wrote {files.config_path}")
        print(f"wrote {files.compose_path}")
        return 0
    if args.bridge_command == "up":
        files = bridge_up(route.display_model, repo_root=args.repo_root, env=env)
        print(f"bridge running from {files.runtime_dir}")
        return 0
    raise AssertionError(args.bridge_command)


def _web(args: argparse.Namespace) -> int:
    from fugue.web import run_web

    run_web(host=args.host, port=args.port)
    return 0


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
            f"dataset:{manifest.dataset.ref}",
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
        env[key.strip()] = value.strip().strip("'\"")
    return env


if __name__ == "__main__":
    raise SystemExit(main())
