from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from fugue.bench.export import (
    export_rows,
    publish_to_weave,
    write_jsonl,
    write_parquet,
)
from fugue.bench.manifest import BenchmarkManifest, load_manifest, write_lock
from fugue.bench.memory import (
    build_artifact,
    clone_repo_at_commit,
    write_condition_instruction,
)
from fugue.bridge import bridge_status, bridge_up, write_bridge_files
from fugue.model_plane import resolve_model_route, select_model
from fugue.preflight import print_preflight, run_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fugue")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build memory artifacts")
    _add_manifest_arg(prepare)
    prepare.add_argument("--conditions", help="Comma-separated condition subset")
    prepare.add_argument("--repo-root", type=Path, default=Path.cwd())

    run = subparsers.add_parser("run", help="Run Harbor matrix")
    _add_manifest_arg(run)
    run.add_argument("--harnesses", help="Comma-separated harness subset")
    run.add_argument("--conditions", help="Comma-separated condition subset")
    run.add_argument("--model", help="Model selector: wandb/..., openai/..., anthropic/...")
    run.add_argument("-k", "--n-attempts", type=int)
    run.add_argument("-n", "--n-concurrent", type=int)
    run.add_argument("-l", "--n-tasks", type=int)
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    run.add_argument("--jobs-dir", type=Path)
    run.add_argument("--repo-root", type=Path, default=Path.cwd())
    run.add_argument("--dry-run", action="store_true")

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

    args = parser.parse_args(argv)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "run":
        return _run(args)
    if args.command == "export":
        return _export(args)
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "bridge":
        return _bridge(args)
    if args.command == "web":
        return _web(args)
    raise AssertionError(args.command)


def _add_manifest_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/pilot.yaml"),
        help="Benchmark manifest path",
    )


def _prepare(args: argparse.Namespace) -> int:
    manifest_path = args.manifest
    manifest = load_manifest(manifest_path)
    conditions = manifest.select_conditions(_csv(args.conditions))
    artifact_root = _resolve(args.repo_root, manifest.artifact_root)
    checkout_root = artifact_root / "_checkouts"
    lock_records = []

    for condition in conditions:
        write_condition_instruction(artifact_root, condition)

    for task in manifest.tasks:
        checkout: Path | None = None
        for condition in conditions:
            if condition == "none":
                artifact = build_artifact(
                    condition=condition,
                    task=task,
                    repo_checkout=Path("."),
                    artifact_root=artifact_root,
                )
            else:
                if checkout is None:
                    checkout = clone_repo_at_commit(task, checkout_root)
                artifact = build_artifact(
                    condition=condition,
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
    manifest = load_manifest(args.manifest)
    harnesses = manifest.select_harnesses(_csv(args.harnesses))
    conditions = manifest.select_conditions(_csv(args.conditions))
    jobs_dir = args.jobs_dir or manifest.jobs_dir
    n_attempts = args.n_attempts or manifest.k
    n_concurrent = args.n_concurrent or manifest.n_concurrent
    env = _load_env(args.env_file)
    env["PYTHONPATH"] = _prepend_path(args.repo_root, env.get("PYTHONPATH"))
    env["FUGUE_MEMORY_DIR"] = _resolve(args.repo_root, manifest.artifact_root).as_posix()
    env["FUGUE_LOCK_PATH"] = _resolve(args.repo_root, manifest.lock_path).as_posix()

    for harness in harnesses:
        selected_model = select_model(args.model, harness.model or manifest.model, env)
        route = resolve_model_route(selected_model, env)
        for condition in conditions:
            env_for_run = env | {
                "FUGUE_CONDITION": condition,
                "FUGUE_MODEL": route.display_model,
            }
            job_name = f"pilot-{harness.name}-{condition}"
            cmd = _harbor_run_command(
                manifest=manifest,
                harness=harness.name,
                agent=harness.agent,
                model=route.display_model,
                condition=condition,
                jobs_dir=jobs_dir,
                job_name=job_name,
                n_attempts=n_attempts,
                n_concurrent=n_concurrent,
                n_tasks=args.n_tasks,
                repo_root=args.repo_root,
                env_file=args.env_file,
            )
            print("+ " + " ".join(shlex.quote(part) for part in cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True, env=env_for_run)
    return 0


def _harbor_run_command(
    *,
    manifest: BenchmarkManifest,
    harness: str,
    agent: str,
    model: str,
    condition: str,
    jobs_dir: Path,
    job_name: str,
    n_attempts: int,
    n_concurrent: int,
    n_tasks: int | None,
    repo_root: Path,
    env_file: Path | None,
) -> list[str]:
    cmd = [
        "harbor",
        "run",
        "-d",
        manifest.dataset.harbor_ref,
        "-a",
        agent,
        "-m",
        model,
        "-o",
        jobs_dir.as_posix(),
        "--job-name",
        job_name,
        "-k",
        str(n_attempts),
        "-n",
        str(n_concurrent),
        "--agent-include-logs",
        "**/*",
    ]
    if env_file is not None:
        cmd.extend(["--env-file", env_file.as_posix()])
    for task in manifest.tasks:
        cmd.extend(["-i", task.id])
    if n_tasks is not None:
        cmd.extend(["-l", str(n_tasks)])
    instruction = write_condition_instruction(
        _resolve(repo_root, manifest.artifact_root), condition
    )
    if instruction is not None:
        cmd.extend(["--extra-instruction-path", instruction.as_posix()])
    return cmd


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


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


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


def _prepend_path(path: Path, existing: str | None) -> str:
    root = path.resolve().as_posix()
    return root if not existing else f"{root}:{existing}"


if __name__ == "__main__":
    raise SystemExit(main())
