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
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from filelock import FileLock

from fugue.bench.context import (
    DEFAULT_CACHE_ROOT,
    ContextRuntime,
    RepositorySnapshot,
    TrialContext,
    bind_context,
    get_context_system,
    prepare_context,
)
from fugue.bench.execution import latest_cell_records, read_run_manifest
from fugue.bench.library import ExperimentSpec, get_prompt, get_skill

if TYPE_CHECKING:
    from fugue.bench.job_config import RenderedJob


INPUT_LOCK_NAME = "input-lock.json"
DEPLOYMENTS_DIR = Path(".fugue/runtime/deployments")
SERVE_CONTEXT_SYSTEMS = frozenset(
    {
        "none",
        "agentsmd",
        "openwiki",
        "aider-repomap",
        "rag-bm25",
        "rag-dense",
        "rag-hybrid",
    }
)
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
_TERMINAL_RUN_STATES = {"passed", "failed", "cancelled", "interrupted"}
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


def write_run_input_lock(
    repo_root: Path,
    run_id: str,
    experiment: ExperimentSpec,
    jobs: list[RenderedJob],
    *,
    env: dict[str, str] | None = None,
) -> Path:
    """Persist replay-affecting run inputs once, without serializing secrets."""
    secret_names = {
        value: name
        for name, value in (env or {}).items()
        if _SENSITIVE_NAME.search(name) and len(value) >= 8
    }
    candidates: dict[str, dict[str, Any]] = {}
    variants = {variant.id: variant for variant in experiment.variants}
    for job in jobs:
        meta = dict(job.config.get("fugue") or {})
        variant = variants.get(job.variant_id)
        if variant is None:
            raise ValueError(f"rendered job references unknown variant: {job.variant_id}")
        context_spec = get_context_system(job.context_system_id, repo_root)
        context_source_sha256 = _sha256_json(
            {**context_spec.to_dict(), "path": None}
        )
        if variant.context.config:
            context_spec = replace(
                context_spec,
                config=_merge_dicts(context_spec.config, variant.context.config),
            )
        prompt_ids = [job.prompt_id] if job.prompt_id else []
        prompt_assets = {
            item_id: _asset_record(get_prompt(item_id, repo_root))
            for item_id in prompt_ids
        }
        skill_assets = {
            item_id: _asset_record(get_skill(item_id, repo_root))
            for item_id in job.skill_ids
        }
        required_env = {job.route.api_key_env, "WANDB_API_KEY"}
        required_env.update(context_spec.required_env)
        agent = _portable_config(
            dict((job.config.get("agents") or [{}])[0]),
            required_env,
            secret_names,
        )
        agent.pop("skills", None)
        environment_source = dict(job.config.get("environment") or {})
        # Task-specific prepared-context mounts and generated compose files are
        # replay coordinates, not candidate identity. Packaging prepares and
        # binds the chosen context again against the production workspace.
        environment_source.pop("mounts", None)
        environment_source.pop("extra_docker_compose", None)
        environment = _portable_config(
            environment_source, required_env, secret_names
        )
        candidate = {
            "candidate_id": job.candidate_id,
            "experiment_id": experiment.id,
            "harness": job.harness,
            "variant_id": job.variant_id,
            "variant_label": job.variant_label,
            "model_provider": job.route.provider,
            "model": job.route.display_model,
            "model_route": asdict(job.route),
            "model_api_key_env": job.route.api_key_env,
            "trace_content": experiment.trace_content,
            "context": {
                **context_spec.to_dict(),
                "path": None,
            },
            "context_config_hash": meta.get("context_config_hash"),
            "context_source_sha256": context_source_sha256,
            "agent_config_hash": job.agent_config_hash,
            "content_hashes": meta.get("content_hashes") or {},
            "prompt_assets": prompt_assets,
            "skill_assets": skill_assets,
            "agent": agent,
            "environment": environment,
            "required_env": sorted(name for name in required_env if name),
        }
        candidate["configuration_sha256"] = _sha256_json(candidate)
        existing = candidates.get(job.candidate_id)
        if existing is not None and existing != candidate:
            raise ValueError(
                f"candidate {job.candidate_id} rendered with inconsistent inputs"
            )
        candidates[job.candidate_id] = candidate

    experiment_required_env: set[str] = set()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment": _portable_config(
            experiment.to_dict(), experiment_required_env, secret_names
        ),
        "candidates": candidates,
    }
    payload["lock_sha256"] = _sha256_json(payload)
    _assert_no_secret_values(payload, env or {})
    path = repo_root / ".fugue/runtime" / run_id / INPUT_LOCK_NAME
    _write_immutable_json(path, payload)
    return path


def package_candidate(
    *,
    repo_root: Path,
    run_id: str,
    candidate_id: str,
    workspace: Path,
    image: str,
    platform: str = "linux/amd64",
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
    status = str(manifest.get("status") or "unknown")
    if status not in _TERMINAL_RUN_STATES:
        raise ValueError(f"run {run_id} is not terminal: {status}")

    lock_path = run_dir / INPUT_LOCK_NAME
    lock = _read_json(lock_path, label="run input lock")
    if lock.get("lock_sha256") != _sha256_json(
        {key: value for key, value in lock.items() if key != "lock_sha256"}
    ):
        raise ValueError(f"run input lock is corrupt: {lock_path}")
    candidates = lock.get("candidates") or {}
    candidate = candidates.get(candidate_id) if isinstance(candidates, dict) else None
    if not isinstance(candidate, dict):
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
    _validate_candidate_records(repo_root, candidate, records)
    _validate_locked_assets(repo_root, candidate)

    context = dict(candidate.get("context") or {})
    context_id = str(context.get("id") or "")
    if context_id not in SERVE_CONTEXT_SYSTEMS:
        raise ValueError(f"context system is not approved for serving: {context_id}")
    if "serve" not in set(context.get("capabilities") or []):
        raise ValueError(f"context system does not advertise serve capability: {context_id}")

    workspace_meta, tracked = _tracked_workspace(workspace, env or {})
    runtime_source = _runtime_source_metadata(repo_root)
    runtime_versions = _runtime_versions(repo_root)
    identity = {
        "schema_version": 1,
        "source_run_id": run_id,
        "candidate_id": candidate_id,
        "candidate_configuration_sha256": candidate["configuration_sha256"],
        "input_lock_sha256": lock["lock_sha256"],
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
                    "experiment_id": candidate["experiment_id"],
                    "harness": candidate["harness"],
                    "model_provider": candidate["model_provider"],
                    "model": candidate["model"],
                    "variant_id": candidate["variant_id"],
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
                        "input_lock_sha256": lock["lock_sha256"],
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
        "variant_id": candidate.get("variant_id"),
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
    for kind, getter in (("prompt", get_prompt), ("skill", get_skill)):
        assets = candidate.get(f"{kind}_assets") or {}
        for item_id, locked in assets.items():
            current = getter(item_id, repo_root)
            if current.sha256 != locked.get("sha256") or current.body != locked.get("body"):
                raise ValueError(f"{kind} asset changed since the run: {item_id}")
    context = candidate.get("context") or {}
    current = get_context_system(str(context.get("id") or ""), repo_root)
    source_sha256 = _sha256_json({**current.to_dict(), "path": None})
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
        data = os.readlink(source).encode() if mode == "120000" else source.read_bytes()
        if _sensitive_workspace_path(path) or any(secret in data for secret in secrets):
            raise ValueError(f"tracked workspace file may contain a secret: {relative}")
        digest.update(f"{mode} {relative}\0".encode())
        digest.update(data)
        digest.update(b"\0")
        tracked.append((mode, relative))
    return (
        {
            "commit": _git(workspace, "rev-parse", "HEAD").strip(),
            "remote": _optional_git(workspace, "remote", "get-url", "origin"),
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
    digest = hashlib.sha256()
    files = _runtime_source_files(repo_root)
    for relative in files:
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update((repo_root / relative).read_bytes())
        digest.update(b"\0")
    return {
        "commit": _optional_git(repo_root, "rev-parse", "HEAD"),
        "digest": digest.hexdigest(),
        "files": len(files),
    }


def _runtime_source_files(repo_root: Path) -> list[Path]:
    files = [Path("pyproject.toml"), Path("uv.lock"), Path("README.md")]
    files.extend(
        path.relative_to(repo_root)
        for path in (repo_root / "fugue").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    files.extend(
        path.relative_to(repo_root)
        for path in (repo_root / "configs/fugue/context-systems").glob("*.yaml")
    )
    return sorted(set(files), key=lambda item: item.as_posix())


def _runtime_versions(repo_root: Path) -> dict[str, str]:
    project = tomllib.loads((repo_root / "pyproject.toml").read_text())["project"]
    return {
        "python": PYTHON_RUNTIME_VERSION,
        "harbor": HARBOR_RUNTIME_VERSION,
        "uv": UV_RUNTIME_VERSION,
        "fugue": str(project["version"]),
    }


def _write_assets(candidate: dict[str, Any], destination: Path) -> None:
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
                experiment_id=str(candidate["experiment_id"]),
                workload_id="serve",
                task_id="production-workspace",
                harness=str(candidate["harness"]),
            ),
            runtime,
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
    skill_ids = sorted((value.get("skill_assets") or {}).keys())
    agent["skills"] = [
        f"/opt/fugue/deployment/assets/skills/{item_id}" for item_id in skill_ids
    ]
    context_id = str((value.get("context") or {}).get("id") or "none")
    if context_id.startswith("rag-"):
        existing_servers = [
            item
            for item in (agent.get("mcp_servers") or [])
            if item.get("name") != "fugue-context"
        ]
        agent["mcp_servers"] = [
            *existing_servers,
            {
                "name": "fugue-context",
                "transport": "stdio",
                "command": "python",
                "args": [
                    "/fugue-src/fugue/mcp_proxy.py",
                    "--name",
                    "fugue-context",
                    "--cwd",
                    "/workspace",
                    "--",
                    "python",
                    "-m",
                    "fugue.context_server",
                    "--system",
                    context_id,
                    "--prepared",
                    "/fugue-context",
                ],
            }
        ]
        agent_env = dict(agent.get("env") or {})
        agent_env.pop("FUGUE_CONTEXT_ENDPOINT", None)
        agent_env.update(
            {
                "FUGUE_REPO_ROOT": "/fugue-src",
                "PYTHONPATH": "/fugue-src",
            }
        )
        agent["env"] = agent_env
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
    && apt-get install -y --no-install-recommends ca-certificates curl docker-cli git nodejs npm \\
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


def _portable_config(
    value: Any, required_env: set[str], secret_names: dict[str, str]
) -> Any:
    if isinstance(value, list):
        return [
            _portable_config(item, required_env, secret_names) for item in value
        ]
    if not isinstance(value, dict):
        if isinstance(value, str) and value in secret_names:
            name = secret_names[value]
            required_env.add(name)
            return f"${{{name}}}"
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if (
            key == "env" or str(key).endswith("_env")
        ) and isinstance(item, dict):
            env_values: dict[str, str] = {}
            for name, env_value in item.items():
                name = str(name)
                if _SENSITIVE_NAME.search(name):
                    required_env.add(name)
                    env_values[name] = f"${{{name}}}"
                elif str(env_value) in secret_names:
                    secret_name = secret_names[str(env_value)]
                    required_env.add(secret_name)
                    env_values[name] = f"${{{secret_name}}}"
                else:
                    env_values[name] = str(env_value)
            result[key] = env_values
        else:
            result[str(key)] = _portable_config(
                item, required_env, secret_names
            )
    return result


def _assert_no_secret_values(value: Any, env: dict[str, str]) -> None:
    body = json.dumps(value, sort_keys=True, default=str)
    for name, secret in env.items():
        if _SENSITIVE_NAME.search(name) and len(secret) >= 8 and secret in body:
            raise ValueError(f"refusing to serialize runtime secret: {name}")


def _asset_record(asset: Any) -> dict[str, str]:
    return {"sha256": str(asset.sha256), "body": str(asset.body)}


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _sensitive_workspace_path(path: PurePosixPath) -> bool:
    name = path.name.lower()
    return (
        name in _SENSITIVE_PATH_NAMES
        or name.endswith((".pem", ".p12", ".pfx"))
        or name in {"id_rsa", "id_ed25519"}
    )


def _write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{path}.lock"):
        if path.exists():
            existing = _read_json(path, label=path.name)
            if existing != value:
                raise ValueError(f"immutable runtime input already exists: {path}")
            return
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        _write_json(temp, value)
        os.replace(temp, path)


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
