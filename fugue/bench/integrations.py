from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import yaml

from fugue.bench.library import IntegrationSelection

INTEGRATION_ROOT = Path("configs") / "fugue" / "integrations"
INTEGRATION_LOCK_PATH = Path("configs") / "fugue" / "integrations.lock.yaml"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_IMAGE_DIGEST_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_SUPPORT_LEVELS = {"supported", "experimental", "not_applicable", "disabled"}
_TRANSPORTS = {"stdio", "sse", "streamable-http"}


@dataclass(frozen=True)
class IntegrationRuntime:
    type: str
    image: str | None = None
    service: str | None = None
    port: int | None = None
    url: str | None = None
    command: tuple[str, ...] = ()
    healthcheck: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegrationInterface:
    type: str
    name: str
    transport: str | None = None
    path: str | None = None
    url: str | None = None
    allowed_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntegrationSpec:
    id: str
    version: str
    support: str
    runtime: IntegrationRuntime
    interfaces: tuple[IntegrationInterface, ...]
    capabilities: tuple[str, ...] = ()
    required_env: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    config_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class IntegrationBinding:
    ids: tuple[str, ...] = ()
    mcp_servers: tuple[dict[str, Any], ...] = ()
    compose_files: tuple[Path, ...] = ()
    instruction_paths: tuple[Path, ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    allowed_hosts: tuple[str, ...] = ()
    provenance: tuple[dict[str, Any], ...] = ()
    applicable: bool = True
    skip_reason: str | None = None


def load_integration(integration_id: str, repo_root: Path) -> IntegrationSpec:
    _validate_id(integration_id, "integration id")
    path = repo_root / INTEGRATION_ROOT / f"{integration_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"integration not found: {integration_id}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: integration must be a mapping")
    allowed = {
        "id",
        "version",
        "support",
        "runtime",
        "interfaces",
        "capabilities",
        "required_env",
        "allowed_hosts",
        "instructions",
        "artifacts",
        "config_schema",
    }
    _reject_unknown(raw, allowed, path, "integration")
    declared_id = str(raw.get("id") or integration_id)
    if declared_id != integration_id:
        raise ValueError(f"{path}: declared id {declared_id!r} does not match filename")
    version = str(raw.get("version") or "").strip()
    if not version:
        raise ValueError(f"{path}: version is required")
    support = str(raw.get("support") or "experimental")
    if support not in _SUPPORT_LEVELS:
        raise ValueError(f"{path}: unsupported support level {support!r}")
    runtime = _runtime(raw.get("runtime"), path)
    interfaces = _interfaces(raw.get("interfaces"), path)
    required_env = tuple(_string_list(raw.get("required_env")))
    for name in required_env:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError(f"{path}: invalid required environment variable {name!r}")
    allowed_hosts = tuple(_string_list(raw.get("allowed_hosts")))
    for host in allowed_hosts:
        _validate_host(host, path)
    if runtime.type == "external":
        runtime_host = str(urlparse(str(runtime.url)).hostname or "")
        if runtime_host not in allowed_hosts:
            raise ValueError(
                f"{path}: external runtime host {runtime_host!r} must be explicitly "
                "listed in allowed_hosts"
            )
    for interface in interfaces:
        if interface.allowed_tools and interface.transport != "stdio":
            raise ValueError(
                f"{path}: allowed_tools is currently enforceable only for stdio MCP; "
                f"{interface.name} requires a reviewed HTTP policy gateway"
            )
        if interface.url:
            _validate_external_url(interface.url, path)
            interface_host = str(urlparse(interface.url).hostname or "")
            if interface_host not in allowed_hosts:
                raise ValueError(
                    f"{path}: interface host {interface_host!r} must be explicitly "
                    "listed in allowed_hosts"
                )
    instructions = tuple(_string_list(raw.get("instructions")))
    for instruction in instructions:
        _validate_relative_path(instruction, path)
    artifacts_raw = raw.get("artifacts") or []
    if not isinstance(artifacts_raw, list) or not all(
        isinstance(item, dict) for item in artifacts_raw
    ):
        raise ValueError(f"{path}: artifacts must be a list of mappings")
    config_schema = raw.get("config_schema") or {}
    if not isinstance(config_schema, dict):
        raise ValueError(f"{path}: config_schema must be a mapping")
    return IntegrationSpec(
        id=integration_id,
        version=version,
        support=support,
        runtime=runtime,
        interfaces=interfaces,
        capabilities=tuple(_string_list(raw.get("capabilities"))),
        required_env=required_env,
        allowed_hosts=allowed_hosts,
        instructions=instructions,
        artifacts=tuple(dict(item) for item in artifacts_raw),
        config_schema=dict(config_schema),
    )


def list_integrations(repo_root: Path) -> list[IntegrationSpec]:
    root = repo_root / INTEGRATION_ROOT
    if not root.is_dir():
        return []
    return [load_integration(path.stem, repo_root) for path in sorted(root.glob("*.yaml"))]


def bind_integrations(
    selections: list[IntegrationSelection],
    *,
    repo_root: Path,
    runtime_root: Path,
    job_name: str,
    env: dict[str, str],
    write: bool,
) -> IntegrationBinding:
    if not selections:
        return IntegrationBinding()
    ids: list[str] = []
    servers: list[dict[str, Any]] = []
    compose_files: list[Path] = []
    instruction_paths: list[Path] = []
    artifacts: list[dict[str, Any]] = []
    binding_env: dict[str, str] = {}
    allowed_hosts: list[str] = []
    provenance: list[dict[str, Any]] = []
    names: set[str] = set()
    for selection in selections:
        spec = load_integration(selection.id, repo_root)
        _validate_selection_config(spec, selection.config)
        ids.append(spec.id)
        provenance.append(
            {
                "id": spec.id,
                "version": spec.version,
                "support": spec.support,
                "config_hash": spec.config_hash,
                "selection_config_hash": _stable_hash(selection.config),
                "runtime_type": spec.runtime.type,
                "image": spec.runtime.image,
                "allowed_tools": {
                    interface.name: list(interface.allowed_tools)
                    for interface in spec.interfaces
                    if interface.allowed_tools
                },
            }
        )
        if spec.support in {"not_applicable", "disabled"}:
            return IntegrationBinding(
                ids=tuple(ids),
                provenance=tuple(provenance),
                applicable=False,
                skip_reason=f"integration {spec.id} is {spec.support}",
            )
        missing = [name for name in spec.required_env if not env.get(name, "").strip()]
        if missing:
            return IntegrationBinding(
                ids=tuple(ids),
                provenance=tuple(provenance),
                applicable=False,
                skip_reason=(
                    f"integration {spec.id} requires environment: {', '.join(missing)}"
                ),
            )
        # Harbor resolves these from the trial process environment. Never serialize
        # credential values into a generated JobConfig.
        binding_env.update({name: f"${{{name}}}" for name in spec.required_env})
        allowed_hosts.extend(spec.allowed_hosts)
        instruction_paths.extend(
            _instruction_path(repo_root, spec.id, value) for value in spec.instructions
        )
        artifacts.extend(spec.artifacts)
        for interface in spec.interfaces:
            if interface.name in names:
                raise ValueError(f"duplicate integration interface name: {interface.name}")
            names.add(interface.name)
            if interface.type == "http":
                endpoint_name = _endpoint_env_name(spec.id, interface.name)
                if endpoint_name in binding_env:
                    raise ValueError(
                        f"integration endpoint environment collision: {endpoint_name}"
                    )
                binding_env[endpoint_name] = _interface_url(spec, interface)
                continue
            server = _mcp_server(spec, interface)
            if interface.allowed_tools:
                server["fugue_allowed_tools"] = list(interface.allowed_tools)
            servers.append(server)
        if spec.runtime.type == "compose":
            compose_path = runtime_root / "integrations" / f"{job_name}-{spec.id}.yaml"
            if write:
                compose_path.parent.mkdir(parents=True, exist_ok=True)
                compose_path.write_text(
                    yaml.safe_dump(_compose(spec, selection.config), sort_keys=False)
                )
            compose_files.append(compose_path)
    return IntegrationBinding(
        ids=tuple(ids),
        mcp_servers=tuple(servers),
        compose_files=tuple(compose_files),
        instruction_paths=tuple(instruction_paths),
        artifacts=tuple(artifacts),
        env=binding_env,
        allowed_hosts=tuple(dict.fromkeys(allowed_hosts)),
        provenance=tuple(provenance),
    )


def effective_selections(
    experiment: list[IntegrationSelection],
    variant: list[IntegrationSelection] | None,
) -> list[IntegrationSelection]:
    return list(experiment if variant is None else variant)


def _runtime(raw: Any, path: Path) -> IntegrationRuntime:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: runtime must be a mapping")
    allowed = {
        "type",
        "image",
        "service",
        "port",
        "url",
        "command",
        "healthcheck",
        "resources",
    }
    _reject_unknown(raw, allowed, path, "runtime")
    runtime_type = str(raw.get("type") or "")
    if runtime_type not in {"compose", "external", "builtin"}:
        raise ValueError(f"{path}: runtime.type must be compose, external, or builtin")
    image = str(raw["image"]) if raw.get("image") else None
    service = str(raw.get("service") or "") or None
    port = int(raw["port"]) if raw.get("port") is not None else None
    url = str(raw["url"]) if raw.get("url") else None
    if runtime_type == "compose":
        if not image or not _IMAGE_DIGEST_RE.fullmatch(image):
            raise ValueError(f"{path}: compose images must be pinned by sha256 digest")
        if not port or not 1 <= port <= 65535:
            raise ValueError(f"{path}: compose runtime requires a valid port")
        service = service or PurePosixPath(path).stem
        _validate_id(service, "compose service")
    if runtime_type == "external":
        _validate_external_url(url, path)
    command = tuple(_string_list(raw.get("command")))
    healthcheck = raw.get("healthcheck") or {}
    resources = raw.get("resources") or {}
    if not isinstance(healthcheck, dict) or not isinstance(resources, dict):
        raise ValueError(f"{path}: healthcheck and resources must be mappings")
    return IntegrationRuntime(
        type=runtime_type,
        image=image,
        service=service,
        port=port,
        url=url,
        command=command,
        healthcheck=dict(healthcheck),
        resources=dict(resources),
    )


def _interfaces(raw: Any, path: Path) -> tuple[IntegrationInterface, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: interfaces must be a non-empty list")
    result: list[IntegrationInterface] = []
    for index, value in enumerate(raw, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: interface {index} must be a mapping")
        _reject_unknown(
            value,
            {"type", "name", "transport", "path", "url", "allowed_tools"},
            path,
            f"interface {index}",
        )
        interface_type = str(value.get("type") or "")
        if interface_type not in {"mcp", "http"}:
            raise ValueError(f"{path}: interface type must be mcp or http")
        name = _validate_id(str(value.get("name") or ""), "interface name")
        transport = str(value.get("transport") or "streamable-http")
        if interface_type == "mcp" and transport not in _TRANSPORTS:
            raise ValueError(f"{path}: unsupported MCP transport {transport!r}")
        interface_path = str(value["path"]) if value.get("path") else None
        if interface_path and not interface_path.startswith("/"):
            raise ValueError(f"{path}: interface path must start with '/'")
        result.append(
            IntegrationInterface(
                type=interface_type,
                name=name,
                transport=transport if interface_type == "mcp" else None,
                path=interface_path,
                url=str(value["url"]) if value.get("url") else None,
                allowed_tools=tuple(_string_list(value.get("allowed_tools"))),
            )
        )
    names = [item.name for item in result]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate interface names")
    return tuple(result)


def _mcp_server(
    spec: IntegrationSpec, interface: IntegrationInterface
) -> dict[str, Any]:
    runtime = spec.runtime
    if interface.transport == "stdio":
        if runtime.type != "builtin" or not runtime.command:
            raise ValueError(
                f"integration {spec.id}: stdio requires a reviewed builtin command"
            )
        return {
            "name": interface.name,
            "transport": "stdio",
            "command": runtime.command[0],
            "args": list(runtime.command[1:]),
        }
    url = _interface_url(spec, interface)
    if not url:
        raise ValueError(f"integration {spec.id}: MCP interface requires a URL")
    return {
        "name": interface.name,
        "transport": interface.transport,
        "url": url,
    }


def _interface_url(
    spec: IntegrationSpec, interface: IntegrationInterface
) -> str:
    runtime = spec.runtime
    if runtime.type == "compose":
        return f"http://{runtime.service}:{runtime.port}{interface.path or ''}"
    if runtime.type == "external":
        return interface.url or _join_url(str(runtime.url), interface.path)
    if interface.url:
        return interface.url
    raise ValueError(f"integration {spec.id}: {interface.name} requires a URL")


def _endpoint_env_name(integration_id: str, interface_name: str) -> str:
    value = f"FUGUE_INTEGRATION_{integration_id}_{interface_name}_URL"
    return re.sub(r"[^A-Za-z0-9]+", "_", value).upper()


def _compose(spec: IntegrationSpec, config: dict[str, Any]) -> dict[str, Any]:
    runtime = spec.runtime
    assert runtime.service and runtime.image and runtime.port
    service: dict[str, Any] = {
        "image": runtime.image,
        "read_only": True,
        "user": "65532:65532",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "tmpfs": ["/tmp:rw,noexec,nosuid,size=64m"],
        "environment": {
            name: f"${{{name}}}" for name in spec.required_env
        },
        "expose": [str(runtime.port)],
    }
    if runtime.command:
        service["command"] = list(runtime.command)
    if config:
        service["environment"]["FUGUE_INTEGRATION_CONFIG"] = json.dumps(
            config, sort_keys=True, separators=(",", ":")
        )
    if runtime.resources:
        service["deploy"] = {"resources": runtime.resources}
    health = runtime.healthcheck
    if health:
        if health.get("path"):
            health_path = str(health["path"])
            service["healthcheck"] = {
                "test": [
                    "CMD",
                    "python",
                    "-c",
                    (
                        "import urllib.request; "
                        f"urllib.request.urlopen('http://127.0.0.1:{runtime.port}{health_path}', timeout=2)"
                    ),
                ],
                "interval": str(health.get("interval") or "2s"),
                "timeout": str(health.get("timeout") or "3s"),
                "retries": int(health.get("retries") or 30),
            }
        elif health.get("command"):
            service["healthcheck"] = {
                "test": ["CMD", *_string_list(health["command"])],
                "interval": str(health.get("interval") or "2s"),
                "timeout": str(health.get("timeout") or "3s"),
                "retries": int(health.get("retries") or 30),
            }
    depends_on: dict[str, Any] = {
        runtime.service: {
            "condition": "service_healthy" if service.get("healthcheck") else "service_started"
        }
    }
    return {"services": {"main": {"depends_on": depends_on}, runtime.service: service}}


def _validate_selection_config(spec: IntegrationSpec, config: dict[str, Any]) -> None:
    sensitive = sorted(
        name
        for name in config
        if any(
            token in name.lower().replace("-", "_")
            for token in ("api_key", "credential", "password", "secret", "token")
        )
    )
    if sensitive:
        raise ValueError(
            f"integration {spec.id} config may not contain secrets; use required_env "
            f"for: {', '.join(sensitive)}"
        )
    schema = spec.config_schema
    if not schema:
        if config:
            raise ValueError(f"integration {spec.id} does not accept configuration")
        return
    unknown = sorted(set(config) - set(schema))
    if unknown:
        raise ValueError(
            f"integration {spec.id} has unknown config field(s): {', '.join(unknown)}"
        )
    for name, contract in schema.items():
        if not isinstance(contract, dict):
            continue
        if contract.get("required") and name not in config:
            raise ValueError(f"integration {spec.id} requires config field {name}")
        if name not in config:
            continue
        expected = contract.get("type")
        value = config[name]
        types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
        if expected in types and not isinstance(value, types[expected]):
            raise ValueError(
                f"integration {spec.id} config {name} must be {expected}"
            )


def _instruction_path(repo_root: Path, integration_id: str, value: str) -> Path:
    path = repo_root / INTEGRATION_ROOT / integration_id / value
    if not path.is_file():
        raise FileNotFoundError(
            f"integration {integration_id} instruction does not exist: {value}"
        )
    return path


def _validate_external_url(value: str | None, path: Path) -> None:
    parsed = urlparse(value or "")
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"{path}: external runtimes require an HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            f"{path}: external URLs may not contain credentials, queries, or fragments"
        )
    host = parsed.hostname.lower()
    if host in {"localhost", "0.0.0.0"} or host.startswith("127.") or host == "::1":
        raise ValueError(f"{path}: external URLs may not target local addresses")


def _validate_host(value: str, path: Path) -> None:
    if "://" in value or "/" in value or ":" in value:
        raise ValueError(f"{path}: allowed_hosts entries must be hostnames")
    if not re.fullmatch(r"(?:\*\.)?[A-Za-z0-9.-]+", value):
        raise ValueError(f"{path}: invalid allowed host {value!r}")


def _validate_relative_path(value: str, path: Path) -> None:
    selected = PurePosixPath(value)
    if selected.is_absolute() or any(part in {"", ".", ".."} for part in selected.parts):
        raise ValueError(f"{path}: instruction paths must be safe relative paths")


def _join_url(base: str, path: str | None) -> str:
    return base.rstrip("/") + (path if path and path.startswith("/") else f"/{path or ''}")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("expected a string or list")
    return [str(item) for item in value]


def _reject_unknown(raw: dict[str, Any], allowed: set[str], path: Path, kind: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown {kind} field(s): {', '.join(unknown)}")


def _validate_id(value: str, kind: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value
