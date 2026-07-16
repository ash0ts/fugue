from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from filelock import FileLock

SERVICES_RUNTIME_ROOT = Path(".fugue") / "runtime" / "services"
GRAPHITI_SERVICE_ID = "graphiti-neo4j"
GRAPHITI_MANAGED_MARKER = "FUGUE_GRAPHITI_MANAGED_SERVICE_ID"
ManagedServiceState = Literal[
    "not_created",
    "starting",
    "healthy",
    "unhealthy",
    "stopped",
    "unavailable",
]


@dataclass(frozen=True)
class ManagedServicePort:
    host: str
    host_port: int
    container_port: int


@dataclass(frozen=True)
class ManagedServiceHealthCheck:
    command: tuple[str, ...]
    interval: str = "5s"
    timeout: str = "5s"
    retries: int = 30
    start_period: str = "10s"


@dataclass(frozen=True)
class ManagedServiceSpec:
    id: str
    image: str
    container_name: str
    ports: tuple[ManagedServicePort, ...]
    health_check: ManagedServiceHealthCheck
    data_volume: str
    required_env_names: tuple[str, ...]
    credential_env_names: tuple[str, ...]
    host_uri: str
    container_uri: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManagedServiceStatus:
    service_id: str
    state: ManagedServiceState
    ready: bool
    detail: str
    container_name: str
    image: str
    host_uri: str


GRAPHITI_SERVICE = ManagedServiceSpec(
    id=GRAPHITI_SERVICE_ID,
    image=(
        "neo4j:5.26.12-community@"
        "sha256:9f75e8df4325a24f00fdd7a8c0bcce650a58375049b1058e496e8b43d6c36b37"
    ),
    container_name="fugue-graphiti-neo4j",
    ports=(
        ManagedServicePort("127.0.0.1", 7474, 7474),
        ManagedServicePort("127.0.0.1", 7687, 7687),
    ),
    health_check=ManagedServiceHealthCheck(
        command=(
            "CMD-SHELL",
            'cypher-shell -u "$${FUGUE_GRAPHITI_USER}" '
            '-p "$${FUGUE_GRAPHITI_PASSWORD}" '
            "'RETURN 1' >/dev/null 2>&1",
        ),
    ),
    data_volume="fugue-graphiti-neo4j-data",
    required_env_names=(
        "FUGUE_GRAPHITI_URI",
        "FUGUE_GRAPHITI_USER",
        "FUGUE_GRAPHITI_PASSWORD",
    ),
    credential_env_names=("FUGUE_GRAPHITI_USER", "FUGUE_GRAPHITI_PASSWORD"),
    host_uri="bolt://127.0.0.1:7687",
    container_uri="bolt://host.docker.internal:7687",
)

_CONTEXT_SERVICE_SPECS = {"graphiti": GRAPHITI_SERVICE}


def managed_services_for_systems(
    system_ids: Iterable[str],
) -> tuple[ManagedServiceSpec, ...]:
    selected: list[ManagedServiceSpec] = []
    seen: set[str] = set()
    for system_id in system_ids:
        spec = _CONTEXT_SERVICE_SPECS.get(system_id)
        if spec is not None and spec.id not in seen:
            selected.append(spec)
            seen.add(spec.id)
    return tuple(selected)


def managed_service_compose(
    spec: ManagedServiceSpec,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    namespace = _service_namespace(repo_root) if repo_root is not None else ""
    project_name = (
        f"fugue-managed-services-{namespace}" if namespace else "fugue-managed-services"
    )
    data_volume = f"{spec.data_volume}-{namespace}" if namespace else spec.data_volume
    labels = {"io.fugue.managed-service": spec.id}
    if namespace:
        labels["io.fugue.managed-service-namespace"] = namespace
    environment = {
        "FUGUE_GRAPHITI_USER": "${FUGUE_GRAPHITI_USER}",
        "FUGUE_GRAPHITI_PASSWORD": "${FUGUE_GRAPHITI_PASSWORD}",
        "NEO4J_AUTH": "${FUGUE_GRAPHITI_USER}/${FUGUE_GRAPHITI_PASSWORD}",
        "NEO4J_server_memory_heap_initial__size": "256m",
        "NEO4J_server_memory_heap_max__size": "1G",
        "NEO4J_server_memory_pagecache_size": "512m",
    }
    return {
        "name": project_name,
        "services": {
            spec.id: {
                "image": spec.image,
                "container_name": spec.container_name,
                "restart": "unless-stopped",
                "labels": labels,
                "ports": [
                    f"{port.host}:{port.host_port}:{port.container_port}"
                    for port in spec.ports
                ],
                "environment": environment,
                "volumes": [f"{data_volume}:/data"],
                "healthcheck": {
                    "test": list(spec.health_check.command),
                    "interval": spec.health_check.interval,
                    "timeout": spec.health_check.timeout,
                    "retries": spec.health_check.retries,
                    "start_period": spec.health_check.start_period,
                },
            }
        },
        "volumes": {data_volume: {"name": data_volume}},
    }


def managed_service_status(
    spec: ManagedServiceSpec,
    *,
    repo_root: Path | None = None,
) -> ManagedServiceStatus:
    if shutil.which("docker") is None:
        return _status(spec, "unavailable", False, "docker is not installed")
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .State}}\n{{json .Config.Image}}\n{{json .Config.Labels}}",
                spec.container_name,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _status(spec, "unavailable", False, str(exc))
    if result.returncode:
        detail = (result.stderr or result.stdout or "container is not created").strip()
        return _status(spec, "not_created", False, detail)
    lines = result.stdout.splitlines()
    try:
        state = json.loads(lines[0])
        image = json.loads(lines[1])
        labels = json.loads(lines[2])
    except (IndexError, TypeError, json.JSONDecodeError):
        return _status(spec, "unavailable", False, "docker returned invalid state")
    namespace = _service_namespace(repo_root) if repo_root is not None else ""
    if (
        not isinstance(state, dict)
        or image != spec.image
        or not isinstance(labels, dict)
        or labels.get("io.fugue.managed-service") != spec.id
        or (namespace and labels.get("io.fugue.managed-service-namespace") != namespace)
    ):
        return _status(
            spec,
            "unhealthy",
            False,
            "container does not match the pinned managed-service contract",
        )
    if not state.get("Running"):
        detail = str(state.get("Error") or state.get("Status") or "container stopped")
        return _status(spec, "stopped", False, detail)
    health = state.get("Health")
    health_state = str(health.get("Status") or "") if isinstance(health, dict) else ""
    if health_state == "healthy":
        return _status(spec, "healthy", True, "container is healthy")
    if health_state == "unhealthy":
        return _status(spec, "unhealthy", False, "container health check failed")
    return _status(spec, "starting", False, health_state or "health check pending")


def managed_service_statuses(
    specs: Iterable[ManagedServiceSpec],
    *,
    repo_root: Path | None = None,
) -> tuple[ManagedServiceStatus, ...]:
    return tuple(managed_service_status(spec, repo_root=repo_root) for spec in specs)


def start_managed_services(
    specs: Iterable[ManagedServiceSpec],
    *,
    repo_root: Path,
    env: Mapping[str, str],
) -> tuple[ManagedServiceStatus, ...]:
    selected = tuple(specs)
    if selected and shutil.which("docker") is None:
        raise RuntimeError("docker is required to start managed services")
    for spec in selected:
        service_env = _service_environment(spec, repo_root, env, create=True)
        compose_path = _write_compose(spec, repo_root)
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                compose_path.as_posix(),
                "up",
                "-d",
                "--wait",
                "--wait-timeout",
                "120",
            ],
            cwd=repo_root,
            env=service_env,
            check=True,
            timeout=180,
        )
    return managed_service_statuses(selected, repo_root=repo_root)


def stop_managed_services(
    specs: Iterable[ManagedServiceSpec],
    *,
    repo_root: Path,
    env: Mapping[str, str],
) -> tuple[ManagedServiceStatus, ...]:
    selected = tuple(specs)
    if selected and shutil.which("docker") is None:
        raise RuntimeError("docker is required to stop managed services")
    for spec in selected:
        compose_path = _compose_path(spec, repo_root)
        if compose_path.is_file():
            service_env = _service_environment(spec, repo_root, env, create=False)
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    compose_path.as_posix(),
                    "down",
                    "--remove-orphans",
                    "--timeout",
                    "30",
                ],
                cwd=repo_root,
                env=service_env,
                check=True,
                timeout=60,
            )
    return managed_service_statuses(selected, repo_root=repo_root)


def managed_service_environment(
    env: Mapping[str, str],
    *,
    repo_root: Path,
    target: Literal["host", "container"] = "host",
    planning: bool = False,
) -> dict[str, str]:
    values = dict(env)
    uri = values.get("FUGUE_GRAPHITI_URI", "").strip()
    marker = values.get(GRAPHITI_MANAGED_MARKER, "").strip()
    if uri and marker != GRAPHITI_SERVICE.id:
        return values
    credentials = _read_credentials(GRAPHITI_SERVICE, repo_root)
    effective_credentials = {
        name: values.get(name, "").strip() or credentials.get(name, "").strip()
        for name in GRAPHITI_SERVICE.credential_env_names
    }
    if planning:
        effective_credentials = {
            name: value or f"${{{name}}}"
            for name, value in effective_credentials.items()
        }
    if not all(effective_credentials.values()):
        return values
    values.update(effective_credentials)
    values["FUGUE_GRAPHITI_URI"] = (
        GRAPHITI_SERVICE.host_uri
        if target == "host"
        else GRAPHITI_SERVICE.container_uri
    )
    values[GRAPHITI_MANAGED_MARKER] = GRAPHITI_SERVICE.id
    return values


def without_managed_service_environment(env: Mapping[str, str]) -> dict[str, str]:
    values = dict(env)
    if values.get(GRAPHITI_MANAGED_MARKER) != GRAPHITI_SERVICE.id:
        return values
    for name in (*GRAPHITI_SERVICE.required_env_names, GRAPHITI_MANAGED_MARKER):
        values.pop(name, None)
    return values


def _service_environment(
    spec: ManagedServiceSpec,
    repo_root: Path,
    env: Mapping[str, str],
    *,
    create: bool,
) -> dict[str, str]:
    values = dict(env)
    stored = _credentials(spec, repo_root, values, create=create)
    values.setdefault("FUGUE_GRAPHITI_URI", spec.host_uri)
    for name in spec.credential_env_names:
        value = values.get(name, "").strip() or stored.get(name, "").strip()
        if not value:
            raise RuntimeError(f"managed service {spec.id} requires {name}")
        values[name] = value
    return values


def _credentials(
    spec: ManagedServiceSpec,
    repo_root: Path,
    env: Mapping[str, str],
    *,
    create: bool,
) -> dict[str, str]:
    existing = _read_credentials(spec, repo_root)
    missing = [
        name
        for name in spec.credential_env_names
        if not str(env.get(name) or existing.get(name) or "").strip()
    ]
    if not missing or not create:
        return existing
    runtime_dir = _runtime_dir(spec, repo_root)
    runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(runtime_dir, 0o700)
    lock = FileLock(runtime_dir / ".credentials.lock")
    with lock:
        existing = _read_credentials(spec, repo_root)
        generated = dict(existing)
        for name in spec.credential_env_names:
            if str(env.get(name) or generated.get(name) or "").strip():
                continue
            generated[name] = (
                "neo4j" if name == "FUGUE_GRAPHITI_USER" else secrets.token_urlsafe(32)
            )
        _write_credentials(spec, repo_root, generated)
        return generated


def _read_credentials(spec: ManagedServiceSpec, repo_root: Path) -> dict[str, str]:
    path = _credentials_path(spec, repo_root)
    if not path.exists():
        return {}
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"managed service credential path is unsafe: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RuntimeError(f"managed service credentials must use mode 0600: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"managed service credentials are invalid: {path}") from exc
    if not isinstance(raw, dict) or any(
        key not in spec.credential_env_names or not isinstance(value, str)
        for key, value in raw.items()
    ):
        raise RuntimeError(f"managed service credentials are invalid: {path}")
    return {str(key): str(value) for key, value in raw.items()}


def _write_credentials(
    spec: ManagedServiceSpec,
    repo_root: Path,
    values: Mapping[str, str],
) -> None:
    path = _credentials_path(spec, repo_root)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(dict(values), handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_compose(spec: ManagedServiceSpec, repo_root: Path) -> Path:
    runtime_dir = _runtime_dir(spec, repo_root)
    runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = _compose_path(spec, repo_root)
    path.write_text(
        yaml.safe_dump(
            managed_service_compose(spec, repo_root=repo_root),
            sort_keys=False,
        )
    )
    return path


def _runtime_dir(spec: ManagedServiceSpec, repo_root: Path) -> Path:
    root = repo_root.resolve()
    path = repo_root / SERVICES_RUNTIME_ROOT / spec.id
    if not path.resolve(strict=False).is_relative_to(root):
        raise RuntimeError(
            f"managed service runtime must remain inside the repository: {path}"
        )
    return path


def _credentials_path(spec: ManagedServiceSpec, repo_root: Path) -> Path:
    return _runtime_dir(spec, repo_root) / "credentials.json"


def _compose_path(spec: ManagedServiceSpec, repo_root: Path) -> Path:
    return _runtime_dir(spec, repo_root) / "docker-compose.yaml"


def _service_namespace(repo_root: Path | None) -> str:
    if repo_root is None:
        return ""
    return hashlib.sha256(repo_root.resolve().as_posix().encode()).hexdigest()[:12]


def _status(
    spec: ManagedServiceSpec,
    state: ManagedServiceState,
    ready: bool,
    detail: str,
) -> ManagedServiceStatus:
    return ManagedServiceStatus(
        service_id=spec.id,
        state=state,
        ready=ready,
        detail=detail,
        container_name=spec.container_name,
        image=spec.image,
        host_uri=spec.host_uri,
    )
