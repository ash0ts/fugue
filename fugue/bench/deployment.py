from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from filelock import FileLock

from fugue.bench.context import (
    DEFAULT_CACHE_ROOT,
    ContextRuntime,
    RepositorySnapshot,
    TrialContext,
    bind_context,
    context_behavior_digest,
    get_context_system,
    prepare_context,
)
from fugue.bench.execution import latest_cell_records, read_run_manifest
from fugue.bench.library import get_prompt
from fugue.bench.reproducibility import INPUT_LOCK_NAME, verify_snapshot
from fugue.bench.sources import resolve_skill

DEPLOYMENTS_DIR = Path(".fugue/runtime/deployments")
SERVE_PROTOCOLS = ("open-responses", "chat-completions", "ag-ui")
SERVE_PROTOCOL_VERSIONS = {
    "open-responses": "2026-04-24",
    "chat-completions": "openai-compatible",
    "ag-ui": "0.1.19",
}
SERVE_HARNESSES = frozenset({"hermes", "openclaw", "claude-code", "codex"})
DEFAULT_ALLOWED_HOSTS = (
    "api.openai.com",
    "api.anthropic.com",
    "*.wandb.ai",
    "registry.npmjs.org",
    "github.com",
    "objects.githubusercontent.com",
    "deb.debian.org",
    "security.debian.org",
    "pypi.org",
    "files.pythonhosted.org",
)
DEFAULT_RESOURCES = {
    "cpus": 2,
    "memory_mb": 4096,
    "storage_mb": 10240,
    "timeout_sec": 900,
}
PYTHON_RUNTIME_VERSION = "3.13"
HARBOR_RUNTIME_VERSION = "0.18.0"
UV_RUNTIME_VERSION = "0.11.27"
_SENSITIVE_NAME = re.compile(
    r"(?:^|_)(?:api_?key|token|secret|password|credential|private_?key)(?:$|_)",
    re.IGNORECASE,
)
_SENSITIVE_PATH_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "service-account.json",
}


@dataclass(frozen=True)
class DeploymentResult:
    deployment_id: str
    candidate_id: str
    image: str
    path: Path
    spec_path: Path
    workspace_digest: str
    built: bool


def candidate_packageability(
    snapshot: dict[str, Any],
    records: list[dict[str, Any]],
    candidate_id: str,
    *,
    allow_failed: bool = False,
) -> tuple[bool, str]:
    runtime = (snapshot.get("candidate_runtime") or {}).get(candidate_id) or {}
    harness = runtime.get("harness")
    if harness and harness not in SERVE_HARNESSES:
        return False, f"candidate harness is not supported for serving: {harness}"
    planned = [
        item
        for item in snapshot.get("planned_matrix") or []
        if item.get("candidate_id") == candidate_id
    ]
    if not planned:
        return False, "candidate is not present in the planned matrix"
    applicable = [item for item in planned if item.get("applicable") is not False]
    if not applicable:
        return False, "all planned cells are not applicable"
    latest = {str(item.get("cell_id")): item for item in records}
    missing = [item for item in applicable if str(item.get("cell_id")) not in latest]
    if missing:
        return False, f"{len(missing)} planned applicable cell(s) have no durable state"
    statuses = [
        str(latest[str(item.get("cell_id"))].get("status") or "unknown")
        for item in applicable
    ]
    pending = sum(status in {"pending", "running", "starting", "unknown"} for status in statuses)
    if pending:
        return False, f"{pending} planned applicable cell(s) are not terminal"
    passed = sum(
        status == "passed"
        and str(latest[str(item.get("cell_id"))].get("benchmark_outcome"))
        == "passed"
        for item, status in zip(applicable, statuses, strict=True)
    )
    benchmark_failed = sum(
        str(latest[str(item.get("cell_id"))].get("benchmark_outcome"))
        == "failed"
        for item in applicable
    )
    execution_failed = sum(
        status in {"failed", "cancelled", "interrupted"} for status in statuses
    )
    unscored = sum(
        status == "passed"
        and str(latest[str(item.get("cell_id"))].get("benchmark_outcome"))
        not in {"passed", "failed"}
        for item, status in zip(applicable, statuses, strict=True)
    )
    if not passed:
        return False, "candidate has no passed applicable cells"
    failed = benchmark_failed + execution_failed
    if failed and not allow_failed:
        return (
            False,
            "candidate has "
            f"{benchmark_failed} failed benchmark cell(s) and "
            f"{execution_failed} execution failure(s); "
            "pass --allow-failed and confirm",
        )
    if failed:
        unscored_detail = (
            f" and {unscored} unscored terminal cell(s)" if unscored else ""
        )
        return (
            True,
            "packageable with "
            f"{benchmark_failed} failed benchmark cell(s) and "
            f"{execution_failed} execution failure(s) explicitly allowed"
            f"{unscored_detail}",
        )
    if unscored:
        return (
            True,
            f"packageable with {passed} passed and "
            f"{unscored} unscored terminal applicable cell(s)",
        )
    return True, "all planned applicable cells completed and passed"


def package_candidate(
    *,
    repo_root: Path,
    run_id: str,
    candidate_id: str,
    workspace: Path,
    image: str,
    platform: str = "linux/amd64",
    allow_failed: bool = False,
    env: dict[str, str] | None = None,
    build: bool = True,
    runner: Callable[..., Any] = subprocess.run,
) -> DeploymentResult:
    repo_root = repo_root.resolve()
    workspace = workspace.resolve()
    run_dir = repo_root / ".fugue/runtime" / run_id
    manifest = read_run_manifest(run_dir)
    if manifest is None:
        raise FileNotFoundError(f"run not found: {run_id}")
    lock_path = run_dir / INPUT_LOCK_NAME
    lock = _read_json(lock_path, label="run input lock")
    if not verify_snapshot(lock):
        raise ValueError(f"run input lock is corrupt: {lock_path}")
    candidates = lock.get("candidates") or {}
    definition = candidates.get(candidate_id) if isinstance(candidates, dict) else None
    runtimes = lock.get("candidate_runtime") or {}
    candidate = runtimes.get(candidate_id) if isinstance(runtimes, dict) else None
    if not isinstance(definition, dict) or not isinstance(candidate, dict):
        raise ValueError(f"candidate is not present in the run input lock: {candidate_id}")
    if candidate.get("harness") not in SERVE_HARNESSES:
        raise ValueError(
            f"candidate harness is not supported for serving: {candidate.get('harness')}"
        )

    records = [
        item
        for item in latest_cell_records(run_dir / "cells.jsonl")
        if item.get("candidate_id") == candidate_id
    ]
    if not records:
        raise ValueError(f"candidate has no durable run cells: {candidate_id}")
    packageable, reason = candidate_packageability(
        lock, records, candidate_id, allow_failed=allow_failed
    )
    if not packageable:
        raise ValueError(reason)
    _validate_candidate_records(repo_root, candidate, records)
    _validate_locked_assets(repo_root, candidate)

    if candidate.get("integration_ids"):
        raise ValueError(
            "0.1 packaging does not support candidates with integrations"
        )

    context = dict(candidate.get("context") or {})
    context_id = str(context.get("id") or "")
    delivery = str((definition.get("context") or {}).get("delivery") or "portable")
    source_experiment_id = str((lock.get("experiment") or {}).get("id") or "")
    if not source_experiment_id:
        raise ValueError("run input lock is missing its resolved experiment id")
    if delivery not in set(context.get("serve_deliveries") or []):
        raise ValueError(
            f"context system {context_id} has no tested packaged {delivery} delivery"
        )

    workspace_meta, tracked = _tracked_workspace(workspace, env or {})
    runtime_source = _runtime_source_metadata(repo_root)
    runtime_versions = _runtime_versions(repo_root)
    identity = {
        "schema_version": 1,
        "source_run_id": run_id,
        "candidate_id": candidate_id,
        "candidate_configuration_sha256": candidate["configuration_sha256"],
        "input_lock_sha256": lock["snapshot_sha256"],
        "workspace": workspace_meta,
        "runtime_source": runtime_source,
        "runtime_versions": runtime_versions,
        "image": image,
        "platform": platform,
        "protocols": list(SERVE_PROTOCOLS),
        "protocol_versions": SERVE_PROTOCOL_VERSIONS,
        "resources": DEFAULT_RESOURCES,
    }
    deployment_id = _sha256_json(identity)
    deployments_root = repo_root / DEPLOYMENTS_DIR
    deployment_dir = deployments_root / deployment_id
    deployments_root.mkdir(parents=True, exist_ok=True)

    with FileLock(f"{deployment_dir}.lock"):
        if not deployment_dir.exists():
            staging = deployments_root / f".{deployment_id}.{uuid.uuid4().hex}.tmp"
            try:
                staging.mkdir(parents=True)
                _copy_tracked_workspace(workspace, tracked, staging / "workspace")
                _copy_runtime_source(repo_root, staging / "runtime-src")
                _write_assets(candidate, staging / "assets")
                prepared_paths = _prepare_serving_context(
                    repo_root=repo_root,
                    workspace=staging / "workspace",
                    workspace_meta=workspace_meta,
                    candidate=candidate,
                    experiment_id=source_experiment_id,
                    delivery=delivery,
                    destination=staging / "context",
                    env=env or {},
                )
                portable_candidate = _deployment_candidate(
                    candidate, prepared_paths=prepared_paths
                )
                required_env = sorted(
                    set(portable_candidate.get("required_env") or [])
                )
                allowed_hosts = list(DEFAULT_ALLOWED_HOSTS)
                for route_value in (candidate.get("model_route") or {}).values():
                    if isinstance(route_value, str) and "://" in route_value:
                        hostname = urlparse(route_value).hostname
                        if hostname and hostname not in allowed_hosts:
                            allowed_hosts.append(hostname)
                spec = {
                    **identity,
                    "deployment_id": deployment_id,
                    "experiment_id": source_experiment_id,
                    "harness": candidate["harness"],
                    "model_provider": candidate["model_provider"],
                    "model": candidate["model"],
                    "variant_id": f"candidate-{candidate_id[:12]}",
                    "context_system_id": context_id,
                    "context_version": context.get("version"),
                    "context_config_hash": candidate.get("context_config_hash"),
                    "agent_config_hash": candidate.get("agent_config_hash"),
                    "content_hashes": candidate.get("content_hashes") or {},
                    "required_env": required_env,
                    "capabilities": ["text", "candidate-backend-tools"],
                    "network_allowed_hosts": allowed_hosts,
                    "candidate": portable_candidate,
                    "provenance": {
                        "source_run_id": run_id,
                        "candidate_id": candidate_id,
                        "candidate_configuration_sha256": candidate[
                            "configuration_sha256"
                        ],
                        "input_lock_sha256": lock["snapshot_sha256"],
                        "source_experiment_id": source_experiment_id,
                        "source_variant_ids": sorted(
                            {
                                str(item.get("variant_id") or "")
                                for item in records
                                if item.get("variant_id")
                            }
                        ),
                        "workspace": workspace_meta,
                        "runtime_source": runtime_source,
                    },
                }
                _assert_no_secret_values(spec, env or {})
                _write_json(staging / "deployment.json", spec)
                (staging / "Dockerfile").write_text(
                    _dockerfile(spec), encoding="utf-8"
                )
                (staging / ".dockerignore").write_text(
                    "**/__pycache__\n**/*.pyc\n", encoding="utf-8"
                )
                os.replace(staging, deployment_dir)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise

    spec_path = deployment_dir / "deployment.json"
    spec = _read_json(spec_path, label="deployment spec")
    if any(spec.get(key) != value for key, value in identity.items()):
        raise ValueError(
            f"deployment directory does not match its content identity: {deployment_dir}"
        )
    built = False
    if build:
        command = [
            "docker",
            "buildx",
            "build",
            "--load",
            "--platform",
            platform,
            "--tag",
            image,
            deployment_dir.as_posix(),
        ]
        runner(command, check=True, cwd=deployment_dir)
        built = True
    return DeploymentResult(
        deployment_id=deployment_id,
        candidate_id=candidate_id,
        image=image,
        path=deployment_dir,
        spec_path=spec_path,
        workspace_digest=str((spec.get("workspace") or {}).get("digest") or ""),
        built=built,
    )


def _validate_candidate_records(
    repo_root: Path, candidate: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    expected = {
        "harness": candidate.get("harness"),
        "context_system_id": (candidate.get("context") or {}).get("id"),
        "model_provider": candidate.get("model_provider"),
        "model": candidate.get("model"),
    }
    for record in records:
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(
                    f"candidate cell {record.get('cell_id')} disagrees on {key}"
                )
        config_path = Path(str(record.get("config_path") or ""))
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        config = _read_json(config_path, label="candidate job config")
        meta = config.get("fugue") or {}
        for key, expected_value in (
            ("candidate_id", candidate.get("candidate_id")),
            ("context_config_hash", candidate.get("context_config_hash")),
            ("agent_config_hash", candidate.get("agent_config_hash")),
            ("content_hashes", candidate.get("content_hashes")),
        ):
            if meta.get(key) != expected_value:
                raise ValueError(
                    f"candidate cell {record.get('cell_id')} job config disagrees on {key}"
                )


def _validate_locked_assets(repo_root: Path, candidate: dict[str, Any]) -> None:
    for item_id, locked in (candidate.get("prompt_assets") or {}).items():
        current = get_prompt(item_id, repo_root)
        if current.sha256 != locked.get("sha256") or current.body != locked.get("body"):
            raise ValueError(f"prompt asset changed since the run: {item_id}")
    for item_id, locked in (candidate.get("skill_assets") or {}).items():
        current = resolve_skill(item_id, repo_root)
        skill_file = current.path / "SKILL.md" if current.path.is_dir() else current.path
        if (
            current.digest.removeprefix("sha256:") != locked.get("sha256")
            or skill_file.read_text(encoding="utf-8") != locked.get("body")
        ):
            raise ValueError(f"skill asset changed since the run: {item_id}")
    context = candidate.get("context") or {}
    current = get_context_system(str(context.get("id") or ""), repo_root)
    source_sha256 = context_behavior_digest(current)
    if source_sha256 != candidate.get("context_source_sha256"):
        raise ValueError(f"context asset changed since the run: {current.id}")


def _tracked_workspace(
    workspace: Path, env: dict[str, str]
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace does not exist: {workspace}")
    status = _git(workspace, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        raise ValueError("production workspace must be a clean Git checkout")
    raw = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=workspace,
        check=True,
        capture_output=True,
    ).stdout
    tracked: list[tuple[str, str]] = []
    digest = hashlib.sha256()
    secrets = [
        value.encode()
        for name, value in env.items()
        if _SENSITIVE_NAME.search(name) and len(value) >= 8
    ]
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        header, path_bytes = entry.split(b"\t", 1)
        mode = header.split(b" ", 1)[0].decode()
        relative = path_bytes.decode()
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe tracked workspace path: {relative}")
        if mode == "160000":
            raise ValueError(f"tracked submodules are not supported: {relative}")
        source = workspace / relative
        if mode == "120000":
            _validate_symlink(workspace, source, relative)
            data = os.readlink(source).encode()
        else:
            data = source.read_bytes()
        if _sensitive_workspace_path(path) or any(secret in data for secret in secrets):
            raise ValueError(f"tracked workspace file may contain a secret: {relative}")
        digest.update(f"{mode} {relative}\0".encode())
        digest.update(data)
        digest.update(b"\0")
        tracked.append((mode, relative))
    return (
        {
            "commit": _git(workspace, "rev-parse", "HEAD").strip(),
            "remote": _safe_remote(
                _optional_git(workspace, "remote", "get-url", "origin")
            ),
            "digest": digest.hexdigest(),
            "files": len(tracked),
        },
        tracked,
    )


def _copy_tracked_workspace(
    workspace: Path, tracked: list[tuple[str, str]], destination: Path
) -> None:
    destination.mkdir(parents=True)
    for mode, relative in tracked:
        source = workspace / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            target.symlink_to(os.readlink(source))
        else:
            shutil.copyfile(source, target)
            target.chmod(0o755 if mode == "100755" else 0o644)


def _copy_runtime_source(repo_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for relative in _runtime_source_files(repo_root):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / relative, target)


def _runtime_source_metadata(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        raise ValueError("Fugue runtime source must be a clean tracked checkout")
    digest = hashlib.sha256()
    files = _runtime_source_files(repo_root)
    for relative in files:
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update((repo_root / relative).read_bytes())
        digest.update(b"\0")
    return {
        "commit": _optional_git(repo_root, "rev-parse", "HEAD"),
        "remote": _safe_remote(
            _optional_git(repo_root, "remote", "get-url", "origin")
        ),
        "digest": digest.hexdigest(),
        "files": len(files),
    }


def _runtime_source_files(repo_root: Path) -> list[Path]:
    tracked = {
        Path(line)
        for line in _git(repo_root, "ls-files").splitlines()
        if line.strip()
    }
    fixed = {Path("pyproject.toml"), Path("uv.lock"), Path("LICENSE")}
    files = {
        path
        for path in tracked
        if path in fixed
        or (
            len(path.parts) >= 2
            and path.parts[0] == "fugue"
            and path.suffix == ".py"
        )
        or (
            len(path.parts) == 4
            and path.parts[:3] == ("configs", "fugue", "context-systems")
            and path.suffix in {".yaml", ".yml"}
        )
    }
    missing = sorted(path for path in fixed if path not in tracked)
    if missing:
        raise ValueError(
            "required runtime source is not tracked: "
            + ", ".join(path.as_posix() for path in missing)
        )
    for relative in files:
        source = repo_root / relative
        if source.is_symlink():
            raise ValueError(
                f"runtime source allowlist may not contain symlinks: {relative}"
            )
    return sorted(files, key=lambda item: item.as_posix())


def _validate_symlink(root: Path, source: Path, relative: str) -> None:
    target = Path(os.readlink(source))
    if target.is_absolute():
        raise ValueError(f"absolute symlink target is not allowed: {relative}")
    try:
        (source.parent / target).resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"symlink escapes the checkout: {relative}") from exc


def _safe_remote(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.username or parsed.password:
        raise ValueError("credential-bearing Git remote URLs are not allowed")
    if re.search(r"(?:token|password|credential)=", value, re.IGNORECASE):
        raise ValueError("credential-bearing Git remote URLs are not allowed")
    return value


def _runtime_versions(repo_root: Path) -> dict[str, str]:
    project = tomllib.loads((repo_root / "pyproject.toml").read_text())["project"]
    return {
        "python": PYTHON_RUNTIME_VERSION,
        "harbor": HARBOR_RUNTIME_VERSION,
        "uv": UV_RUNTIME_VERSION,
        "fugue": str(project["version"]),
    }


def _write_assets(candidate: dict[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for kind in ("prompt", "skill"):
        for item_id, asset in (candidate.get(f"{kind}_assets") or {}).items():
            path = (
                destination / "prompts" / f"{item_id}.md"
                if kind == "prompt"
                else destination / "skills" / item_id / "SKILL.md"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(asset["body"]), encoding="utf-8")


def _prepare_serving_context(
    *,
    repo_root: Path,
    workspace: Path,
    workspace_meta: dict[str, Any],
    candidate: dict[str, Any],
    experiment_id: str,
    delivery: str,
    destination: Path,
    env: dict[str, str],
) -> list[str]:
    context_data = dict(candidate.get("context") or {})
    spec = get_context_system(str(context_data.get("id") or ""), repo_root)
    spec = replace(spec, config=dict(context_data.get("config") or {}))
    runtime = ContextRuntime(
        repo_root=repo_root,
        cache_root=repo_root / DEFAULT_CACHE_ROOT,
        env=dict(env),
    )
    snapshot = RepositorySnapshot(
        task_id="production-workspace",
        repo=str(workspace_meta.get("remote") or workspace),
        commit=str(workspace_meta["commit"]),
        checkout=workspace,
        dataset_id="fugue-serve",
    )
    prepared = asyncio.run(prepare_context(spec, snapshot, runtime))
    shutil.copytree(prepared.path, destination)
    binding = asyncio.run(
        bind_context(
            spec,
            prepared,
            TrialContext(
                experiment_id=experiment_id,
                workload_id="serve",
                task_id="production-workspace",
                harness=str(candidate["harness"]),
            ),
            runtime,
            delivery=delivery,
        )
    )
    paths: list[str] = []
    for path in binding.extra_instruction_paths:
        try:
            relative = path.resolve().relative_to(prepared.path.resolve())
        except ValueError as exc:
            raise ValueError(
                f"context instruction escapes the prepared artifact: {path}"
            ) from exc
        paths.append(relative.as_posix())
    return paths


def _deployment_candidate(
    candidate: dict[str, Any], *, prepared_paths: list[str]
) -> dict[str, Any]:
    value = json.loads(json.dumps(candidate))
    agent = dict(value.get("agent") or {})
    if agent.get("mcp_servers"):
        raise ValueError(
            "packaged contexts must use a declared MCP-free serving delivery"
        )
    skill_ids = sorted((value.get("skill_assets") or {}).keys())
    agent["skills"] = [
        f"/opt/fugue/deployment/assets/skills/{item_id}" for item_id in skill_ids
    ]
    value["agent"] = agent
    environment = dict(value.get("environment") or {})
    environment.pop("mounts", None)
    environment.pop("extra_docker_compose", None)
    value["environment"] = environment
    value["extra_instruction_paths"] = [
        *[
            f"/opt/fugue/deployment/assets/prompts/{item_id}.md"
            for item_id in sorted((value.get("prompt_assets") or {}).keys())
        ],
        *[f"/opt/fugue/deployment/context/{path}" for path in prepared_paths],
    ]
    return value


def _dockerfile(spec: dict[str, Any]) -> str:
    labels = {
        "org.opencontainers.image.title": "Fugue candidate service",
        "org.opencontainers.image.version": spec["deployment_id"],
        "io.fugue.deployment.id": spec["deployment_id"],
        "io.fugue.source.run-id": spec["source_run_id"],
        "io.fugue.candidate.id": spec["candidate_id"],
        "io.fugue.workspace.digest": spec["workspace"]["digest"],
        "io.fugue.runtime.digest": spec["runtime_source"]["digest"],
        "io.fugue.input-lock.digest": spec["input_lock_sha256"],
        "io.fugue.candidate.configuration-digest": spec[
            "candidate_configuration_sha256"
        ],
        "org.opencontainers.image.revision": spec["workspace"]["commit"],
    }
    label_lines = " \\\n    ".join(f'{key}="{value}"' for key, value in labels.items())
    return f"""FROM python:{PYTHON_RUNTIME_VERSION}-slim

LABEL {label_lines}

RUN apt-get update \\
    && apt-get install -y --no-install-recommends ca-certificates curl docker-buildx docker-cli docker-compose git nodejs npm \\
    && rm -rf /var/lib/apt/lists/*

COPY runtime-src /fugue-src
RUN python -m pip install --no-cache-dir "uv=={UV_RUNTIME_VERSION}" \\
    && cd /fugue-src \\
    && uv sync --frozen --no-dev --extra serve --python /usr/local/bin/python

COPY workspace /workspace
COPY assets /opt/fugue/deployment/assets
COPY context /opt/fugue/deployment/context
COPY deployment.json /opt/fugue/deployment/deployment.json
RUN ln -s /opt/fugue/deployment/context /fugue-context

ENV FUGUE_DEPLOYMENT_SPEC=/opt/fugue/deployment/deployment.json \\
    FUGUE_SERVE_RUNTIME_DIR=/var/lib/fugue/requests \\
    PATH=/fugue-src/.venv/bin:$PATH \\
    PYTHONUNBUFFERED=1
WORKDIR /workspace
EXPOSE 8000
CMD ["python", "-m", "fugue.serve"]
"""


def _assert_no_secret_values(value: Any, env: dict[str, str]) -> None:
    body = json.dumps(value, sort_keys=True, default=str)
    for name, secret in env.items():
        if _SENSITIVE_NAME.search(name) and len(secret) >= 8 and secret in body:
            raise ValueError(f"refusing to serialize runtime secret: {name}")


def _sensitive_workspace_path(path: PurePosixPath) -> bool:
    name = path.name.lower()
    return (
        name in _SENSITIVE_PATH_NAMES
        or name.endswith((".pem", ".p12", ".pfx"))
        or name in {"id_rsa", "id_ed25519"}
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_json(value: Any) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(body).hexdigest()


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _optional_git(workspace: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None
