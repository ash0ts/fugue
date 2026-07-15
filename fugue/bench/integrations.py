from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import yaml

from fugue.bench.library import IntegrationSelection

INTEGRATION_ROOT = Path("configs") / "fugue" / "integrations"
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
    _validate_runtime_interfaces(runtime, interfaces, path)
    required_env = tuple(_string_list(raw.get("required_env")))
    for name in required_env:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError(f"{path}: invalid required environment variable {name!r}")
    if len(set(required_env)) != len(required_env):
        raise ValueError(f"{path}: required_env entries must be unique")
    allowed_hosts = tuple(
        dict.fromkeys(host.lower() for host in _string_list(raw.get("allowed_hosts")))
    )
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
    artifacts = _artifacts(raw.get("artifacts"), path, runtime)
    config_schema = _config_schema(raw.get("config_schema"), path)
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
        artifacts=artifacts,
        config_schema=config_schema,
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
    reserved_ports: dict[int, str] | None = None,
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
    compose_ports: dict[int, str] = dict(reserved_ports or {})
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
            assert spec.runtime.port is not None
            previous = compose_ports.get(spec.runtime.port)
            if previous:
                raise ValueError(
                    f"shared services {previous} and {spec.id} both use "
                    f"port {spec.runtime.port}"
                )
            compose_ports[spec.runtime.port] = spec.id
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
    command = tuple(_string_list(raw.get("command")))
    if any(not item.strip() for item in command):
        raise ValueError(f"{path}: runtime command entries may not be empty")
    healthcheck = _healthcheck(raw.get("healthcheck"), path)
    resources = raw.get("resources") or {}
    if not isinstance(resources, dict):
        raise ValueError(f"{path}: resources must be a mapping")
    if runtime_type == "compose":
        if not image or not _IMAGE_DIGEST_RE.fullmatch(image):
            raise ValueError(f"{path}: compose images must be pinned by sha256 digest")
        if not port or not 1 <= port <= 65535:
            raise ValueError(f"{path}: compose runtime requires a valid port")
        if url:
            raise ValueError(f"{path}: compose runtime may not declare an external URL")
        service = service or PurePosixPath(path).stem
        _validate_id(service, "compose service")
    elif runtime_type == "external":
        _validate_external_url(url, path)
        if any((image, service, port, command, healthcheck, resources)):
            raise ValueError(
                f"{path}: external runtime accepts only type and url fields"
            )
    else:
        if not command:
            raise ValueError(f"{path}: builtin runtime requires a reviewed command")
        if any((image, service, port, url, healthcheck, resources)):
            raise ValueError(
                f"{path}: builtin runtime accepts only type and command fields"
            )
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


def _healthcheck(raw: Any, path: Path) -> dict[str, Any]:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: healthcheck must be a mapping")
    _reject_unknown(
        raw,
        {"path", "command", "interval", "timeout", "retries"},
        path,
        "healthcheck",
    )
    has_path = raw.get("path") is not None
    has_command = raw.get("command") is not None
    if has_path == has_command:
        raise ValueError(
            f"{path}: healthcheck requires exactly one of path or command"
        )
    if has_path and not str(raw["path"]).startswith("/"):
        raise ValueError(f"{path}: healthcheck path must start with '/'")
    if has_command:
        command = _string_list(raw["command"])
        if not command or any(not item.strip() for item in command):
            raise ValueError(f"{path}: healthcheck command may not be empty")
    retries = raw.get("retries", 30)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 1:
        raise ValueError(f"{path}: healthcheck retries must be a positive integer")
    return dict(raw)


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
        if interface_type == "http" and "transport" in value:
            raise ValueError(f"{path}: plain HTTP interfaces may not set transport")
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


def _validate_runtime_interfaces(
    runtime: IntegrationRuntime,
    interfaces: tuple[IntegrationInterface, ...],
    path: Path,
) -> None:
    for interface in interfaces:
        if runtime.type == "compose":
            if interface.transport == "stdio":
                raise ValueError(f"{path}: compose runtime may not use stdio MCP")
            if interface.url:
                raise ValueError(
                    f"{path}: compose interfaces use the declared service and may not set url"
                )
        elif runtime.type == "external":
            if interface.transport == "stdio":
                raise ValueError(f"{path}: external runtime may not use stdio MCP")
        else:
            if interface.type != "mcp" or interface.transport != "stdio":
                raise ValueError(
                    f"{path}: builtin runtime supports only stdio MCP interfaces"
                )
            if interface.path or interface.url:
                raise ValueError(
                    f"{path}: builtin stdio interfaces may not set path or url"
                )


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
        return f"http://127.0.0.1:{runtime.port}{interface.path or ''}"
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
        "network_mode": "service:main",
        "tmpfs": ["/tmp:rw,noexec,nosuid,size=64m"],
        "environment": {
            name: f"${{{name}}}" for name in spec.required_env
        },
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
            health_url = json.dumps(
                f"http://127.0.0.1:{runtime.port}{health_path}"
            )
            service["healthcheck"] = {
                "test": [
                    "CMD",
                    "python",
                    "-c",
                    (
                        "import urllib.request; "
                        f"urllib.request.urlopen({health_url}, timeout=2)"
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
    return {"services": {runtime.service: service}}


def _artifacts(
    raw: Any, path: Path, runtime: IntegrationRuntime
) -> tuple[dict[str, Any], ...]:
    values = raw or []
    if not isinstance(values, list) or not all(
        isinstance(item, dict) for item in values
    ):
        raise ValueError(f"{path}: artifacts must be a list of mappings")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(values, start=1):
        _reject_unknown(
            item,
            {"source", "destination", "exclude", "service"},
            path,
            f"artifact {index}",
        )
        source = str(item.get("source") or "")
        if not source:
            raise ValueError(f"{path}: artifact {index} requires source")
        service = str(item.get("service") or "") or None
        if service and service != "main":
            _validate_id(service, "artifact service")
            if runtime.type != "compose" or service != runtime.service:
                raise ValueError(
                    f"{path}: artifact {index} may target only main or the integration service"
                )
            if not PurePosixPath(source).is_absolute():
                raise ValueError(
                    f"{path}: service artifact {index} requires an absolute source"
                )
        exclude = item.get("exclude") or []
        if not isinstance(exclude, list) or not all(
            isinstance(value, str) for value in exclude
        ):
            raise ValueError(f"{path}: artifact {index} exclude must be a list of strings")
        result.append(dict(item))
    return tuple(result)


def _config_schema(raw: Any, path: Path) -> dict[str, Any]:
    schema = raw or {}
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: config_schema must be a mapping")
    for name, contract in schema.items():
        _validate_id(str(name), "integration config field")
        if _sensitive_name(str(name)):
            raise ValueError(
                f"{path}: secret-like config field {name!r} must use required_env"
            )
        if not isinstance(contract, dict):
            raise ValueError(f"{path}: config schema {name} must be a mapping")
        _reject_unknown(
            contract,
            {"type", "required"},
            path,
            f"config schema {name}",
        )
        expected = contract.get("type")
        if expected not in {"string", "integer", "number", "boolean"}:
            raise ValueError(
                f"{path}: config schema {name} has unsupported type {expected!r}"
            )
        if "required" in contract and not isinstance(contract["required"], bool):
            raise ValueError(f"{path}: config schema {name} required must be boolean")
    return dict(schema)


def _validate_selection_config(spec: IntegrationSpec, config: dict[str, Any]) -> None:
    sensitive = sorted(
        name for name in config if _sensitive_name(name)
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
        if contract.get("required") and name not in config:
            raise ValueError(f"integration {spec.id} requires config field {name}")
        if name not in config:
            continue
        expected = contract.get("type")
        value = config[name]
        valid = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }[str(expected)]
        if not valid:
            raise ValueError(
                f"integration {spec.id} config {name} must be {expected}"
            )


def _sensitive_name(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    return any(
        token in normalized
        for token in ("api_key", "credential", "password", "secret", "token")
    )


def _instruction_path(repo_root: Path, integration_id: str, value: str) -> Path:
    root = repo_root / INTEGRATION_ROOT / integration_id
    path = root / value
    cursor = root
    if root.is_symlink():
        raise ValueError(
            f"integration {integration_id} instruction directory may not be a symlink"
        )
    for part in PurePosixPath(value).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(
                f"integration {integration_id} instruction may not use symlinks: {value}"
            )
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
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"{path}: external URLs may not target local addresses")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError(
            f"{path}: external URLs may not target non-public IP addresses"
        )


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
