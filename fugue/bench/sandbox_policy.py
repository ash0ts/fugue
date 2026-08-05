from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from fugue.bench.candidates import stable_digest

SANDBOX_POLICY_VERSION = 1
_IMAGE_DIGEST = re.compile(r"(?:@sha256:|^sha256:)[0-9a-f]{64}$")
_FORBIDDEN_SERVICE_KEYS = {
    "cap_add",
    "devices",
    "ipc",
    "pid",
    "privileged",
}
_ALLOWED_SERVICE_KEYS = {
    "cap_drop",
    "command",
    "deploy",
    "environment",
    "healthcheck",
    "image",
    "network_mode",
    "pids_limit",
    "pull_policy",
    "read_only",
    "security_opt",
    "tmpfs",
    "user",
    "volumes",
}
_ALLOWED_MOUNT_KEYS = {"bind", "image", "type", "source", "target", "read_only"}
_MEMORY_LIMIT = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+)?[kmgtpe]?(?:i?b)?$", re.I)
_HARBOR_TASK_CPU_LIMIT = "${CPUS:-8.0}"
_HARBOR_TASK_MEMORY_LIMIT = "${MEMORY:-16384M}"
_LOCKED_BRIDGE = "http://host.docker.internal:4000"


@dataclass(frozen=True)
class EffectiveSandboxAttestationV1:
    schema_version: int
    policy_version: int
    source: str
    bridge_endpoint: str | None
    compose_assets: tuple[dict[str, Any], ...]
    services: tuple[dict[str, Any], ...]
    policy_digest: str
    attestation_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["compose_assets"] = list(self.compose_assets)
        value["services"] = list(self.services)
        return value


def attest_harbor_job(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    bridge_required: bool,
    require_files: bool,
    strict_images: bool = False,
) -> EffectiveSandboxAttestationV1:
    environment = config.get("environment") or {}
    if not isinstance(environment, Mapping):
        raise ValueError("Harbor environment must be an object")
    _validate_environment(environment, repo_root, strict_sources=strict_images)
    paths = environment.get("extra_docker_compose") or ()
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
        raise ValueError("Harbor extra_docker_compose must be a list")
    assets: list[dict[str, Any]] = []
    services: list[dict[str, Any]] = []
    for raw_path in paths:
        path = _locked_path(raw_path, repo_root)
        if not path.is_file():
            if require_files:
                raise ValueError(f"locked Harbor Compose fragment is missing: {path}")
            continue
        body = path.read_text(encoding="utf-8")
        document = yaml.safe_load(body)
        selected = _validate_compose(
            document,
            repo_root=repo_root,
            bridge_required=bridge_required,
            strict_images=strict_images,
        )
        assets.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": hashlib.sha256(body.encode()).hexdigest(),
            }
        )
        for item in selected:
            item["compose_path"] = path.relative_to(repo_root).as_posix()
        services.extend(selected)
    policy = {
        "policy_version": SANDBOX_POLICY_VERSION,
        "bridge_required": bridge_required,
        "allowed_network_mode": "service:main",
        "pull_policy": "never",
        "rootless_required_for_non_local": True,
    }
    attestation = EffectiveSandboxAttestationV1(
        schema_version=1,
        policy_version=SANDBOX_POLICY_VERSION,
        source="rendered_harbor_config",
        bridge_endpoint=(
            "http://host.docker.internal:4000" if bridge_required else None
        ),
        compose_assets=tuple(sorted(assets, key=lambda item: item["path"])),
        services=tuple(
            sorted(services, key=lambda item: (item["compose_path"], item["service"]))
        ),
        policy_digest=stable_digest(policy),
    )
    return EffectiveSandboxAttestationV1(
        **{
            **asdict(attestation),
            "attestation_digest": stable_digest(
                {
                    key: value
                    for key, value in attestation.to_dict().items()
                    if key != "attestation_digest"
                }
            ),
        }
    )


def verify_harbor_job_attestation(config_path: Path, repo_root: Path) -> None:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Harbor job config must be an object")
    fugue = raw.get("fugue")
    expected = fugue.get("sandbox_attestation") if isinstance(fugue, Mapping) else None
    if not isinstance(expected, Mapping):
        raise ValueError("Harbor job has no locked sandbox attestation")
    bridge_required = bool(expected.get("bridge_endpoint"))
    observed = attest_harbor_job(
        raw,
        repo_root=repo_root,
        bridge_required=bridge_required,
        require_files=True,
        strict_images=True,
    ).to_dict()
    if observed != dict(expected):
        raise ValueError(
            "effective Harbor Compose inputs differ from the locked sandbox policy"
        )


def _validate_environment(
    environment: Mapping[str, Any],
    repo_root: Path,
    *,
    strict_sources: bool,
) -> None:
    for key in (
        "privileged",
        "devices",
        "cap_add",
        "pid",
        "ipc",
        "network_mode",
        "docker_socket",
    ):
        if key in environment:
            raise ValueError(f"Harbor environment may not set {key}")
    mounts = environment.get("mounts") or ()
    if not isinstance(mounts, Sequence) or isinstance(mounts, (str, bytes)):
        raise ValueError("Harbor environment mounts must be a list")
    for mount in mounts:
        _validate_mount(
            mount,
            repo_root=repo_root,
            service="main",
            strict_sources=strict_sources,
        )


def _validate_compose(
    raw: Any,
    *,
    repo_root: Path,
    bridge_required: bool,
    strict_images: bool,
) -> list[dict[str, Any]]:
    if not isinstance(raw, Mapping) or set(raw) != {"services"}:
        raise ValueError("Harbor Compose fragments may contain only services")
    raw_services = raw["services"]
    if not isinstance(raw_services, Mapping) or not raw_services:
        raise ValueError("Harbor Compose fragment services must be a non-empty object")
    result: list[dict[str, Any]] = []
    for service_name, value in raw_services.items():
        if not isinstance(service_name, str) or not isinstance(value, Mapping):
            raise ValueError("Harbor Compose services must be named objects")
        result.append(
            _validate_service(
                service_name,
                value,
                repo_root=repo_root,
                bridge_required=bridge_required,
                strict_images=strict_images,
            )
        )
    return result


def _validate_service(
    service_name: str,
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    bridge_required: bool,
    strict_images: bool,
) -> dict[str, Any]:
    forbidden = sorted(_FORBIDDEN_SERVICE_KEYS.intersection(value))
    if forbidden:
        raise ValueError(
            f"Harbor service {service_name} sets forbidden options: "
            + ", ".join(forbidden)
        )
    unknown = sorted(set(value) - _ALLOWED_SERVICE_KEYS)
    if unknown:
        raise ValueError(
            f"Harbor service {service_name} sets unsupported options: "
            + ", ".join(unknown)
        )
    if value.get("pull_policy") != "never":
        raise ValueError(f"Harbor service {service_name} must use pull_policy=never")
    if value.get("network_mode") == "host":
        raise ValueError(f"Harbor service {service_name} may not use the host network")
    if set(value.get("cap_drop") or ()) != {"ALL"}:
        raise ValueError(f"Harbor service {service_name} must drop all capabilities")
    security = set(value.get("security_opt") or ())
    if {"seccomp=unconfined", "apparmor=unconfined"}.intersection(security):
        raise ValueError(f"Harbor service {service_name} disables a security profile")
    if not {"no-new-privileges", "no-new-privileges:true"}.intersection(security):
        raise ValueError(
            f"Harbor service {service_name} must enable no-new-privileges"
        )
    _validate_resource_limits(value, service_name)
    pids_limit = _validate_pid_limit(value, service_name)
    if service_name != "main":
        _validate_sidecar(value, service_name, strict_images=strict_images)
    for mount in value.get("volumes") or ():
        _validate_mount(
            mount,
            repo_root=repo_root,
            service=service_name,
            strict_sources=strict_images,
        )
    _validate_bridge(value, bridge_required=bridge_required)
    return {
        "compose_path": "",
        "service": service_name,
        "image": value.get("image"),
        "user": value.get("user"),
        "read_only": value.get("read_only"),
        "network_mode": value.get("network_mode"),
        "cap_drop": list(value.get("cap_drop") or ()),
        "security_opt": list(value.get("security_opt") or ()),
        "pids_limit": pids_limit,
        "limits": (value.get("deploy") or {}).get("resources", {}),
    }


def _validate_pid_limit(value: Mapping[str, Any], service_name: str) -> int:
    deploy = value.get("deploy")
    resources = deploy.get("resources") if isinstance(deploy, Mapping) else None
    limits = resources.get("limits") if isinstance(resources, Mapping) else None
    deploy_pids = limits.get("pids") if isinstance(limits, Mapping) else None
    pids_limit = value.get("pids_limit")
    if pids_limit is not None and deploy_pids is not None:
        raise ValueError(
            f"Harbor service {service_name} may declare only one PID limit"
        )
    pids_limit = pids_limit if pids_limit is not None else deploy_pids
    if (
        not isinstance(pids_limit, int)
        or isinstance(pids_limit, bool)
        or not 0 < pids_limit <= 4096
    ):
        raise ValueError(
            f"Harbor service {service_name} requires a positive PID limit"
        )
    return pids_limit


def _validate_sidecar(
    value: Mapping[str, Any],
    service_name: str,
    *,
    strict_images: bool,
) -> None:
    image = str(value.get("image") or "")
    if strict_images and not _IMAGE_DIGEST.search(image):
        raise ValueError(
            f"Harbor service {service_name} image must be pinned by sha256"
        )
    if value.get("network_mode") != "service:main":
        raise ValueError(
            f"Harbor service {service_name} must share only main's network"
        )
    if value.get("read_only") is not True:
        raise ValueError(
            f"Harbor service {service_name} must use a read-only root filesystem"
        )
    user = str(value.get("user") or "")
    if not user or user.split(":", 1)[0] in {"0", "root"}:
        raise ValueError(f"Harbor service {service_name} must run as non-root")


def _validate_bridge(value: Mapping[str, Any], *, bridge_required: bool) -> None:
    text = json.dumps(value, sort_keys=True)
    contains_bridge = "host.docker.internal" in text
    if contains_bridge and not bridge_required:
        raise ValueError(
            "host.docker.internal is allowed only for a candidate with a locked bridge"
        )
    if contains_bridge and any(
        item != _LOCKED_BRIDGE
        for item in _string_values(value)
        if "host.docker.internal" in item
    ):
        raise ValueError("Harbor bridge use must match the locked endpoint exactly")


def _validate_resource_limits(service: Mapping[str, Any], service_name: str) -> None:
    deploy = service.get("deploy")
    resources = deploy.get("resources") if isinstance(deploy, Mapping) else None
    limits = resources.get("limits") if isinstance(resources, Mapping) else None
    if (
        not isinstance(limits, Mapping)
        or not limits.get("cpus")
        or not limits.get("memory")
    ):
        raise ValueError(
            f"Harbor service {service_name} requires CPU and memory limits"
        )
    permitted = {"cpus", "memory", "pids"}
    if not {"cpus", "memory"} <= set(limits) or set(limits) - permitted:
        raise ValueError(
            f"Harbor service {service_name} permits only CPU, memory, and PID limits"
        )
    cpu_limit = str(limits["cpus"])
    if service_name != "main" or cpu_limit != _HARBOR_TASK_CPU_LIMIT:
        try:
            cpus = float(cpu_limit)
        except ValueError as exc:
            raise ValueError(
                f"Harbor service {service_name} has an invalid CPU limit"
            ) from exc
        if not 0 < cpus <= 64:
            raise ValueError(f"Harbor service {service_name} has an invalid CPU limit")
    memory_limit = str(limits["memory"])
    if service_name == "main" and memory_limit == _HARBOR_TASK_MEMORY_LIMIT:
        return
    if not _MEMORY_LIMIT.fullmatch(memory_limit):
        raise ValueError(f"Harbor service {service_name} has an invalid memory limit")


def _validate_mount(
    raw: Any,
    *,
    repo_root: Path,
    service: str,
    strict_sources: bool,
) -> None:
    if isinstance(raw, str):
        source, separator, remainder = raw.partition(":")
        target, _, mode = remainder.partition(":")
        read_only = mode == "ro"
    elif isinstance(raw, Mapping):
        if raw.get("type") == "image":
            _validate_image_mount(raw, service=service)
            return
        bind = raw.get("bind") or {}
        if (
            set(raw) - _ALLOWED_MOUNT_KEYS
            or raw.get("type") != "bind"
            or not isinstance(bind, Mapping)
            or set(bind) - {"create_host_path"}
            or bind.get("create_host_path") not in {None, False}
        ):
            raise ValueError(
                f"Harbor service {service} permits only locked bind mounts"
            )
        source = str(raw.get("source") or "")
        target = str(raw.get("target") or "")
        read_only = raw.get("read_only") is True
    else:
        raise ValueError(f"Harbor service {service} has an invalid mount")
    if not source or not target:
        raise ValueError(f"Harbor service {service} has an incomplete mount")
    if not Path(target).is_absolute() or ".." in Path(target).parts:
        raise ValueError(f"Harbor service {service} mount targets must be absolute")
    if "docker.sock" in source or "docker.sock" in target:
        raise ValueError("Harbor cells may not mount the Docker socket")
    source_path = Path(source)
    if not source_path.is_absolute():
        raise ValueError("Harbor bind sources must be absolute")
    resolved = source_path.resolve()
    root = repo_root.resolve()
    package_root = Path(__file__).resolve().parents[2]
    reviewed_package_mount = (
        read_only
        and resolved.is_relative_to(package_root)
        and (
            target.startswith("/fugue-src/") or target == "/usr/local/bin/fugue-context"
        )
    )
    if not resolved.is_relative_to(root) and not reviewed_package_mount:
        raise ValueError("Harbor bind sources must stay within the Fugue checkout")
    if strict_sources and not source_path.exists():
        raise ValueError("Harbor bind sources must exist before execution")
    writable_runtime = resolved.is_relative_to(root / ".fugue" / "runtime")
    if not read_only and not writable_runtime:
        raise ValueError(
            "Harbor source and task input mounts must be read-only; only dedicated "
            "Fugue runtime evidence paths may be writable"
        )


def _validate_image_mount(
    raw: Mapping[str, Any],
    *,
    service: str,
) -> None:
    image = raw.get("image")
    if (
        set(raw) - _ALLOWED_MOUNT_KEYS
        or not isinstance(image, Mapping)
        or set(image) != {"subpath"}
        or raw.get("bind") is not None
        or raw.get("read_only") is not True
    ):
        raise ValueError(
            f"Harbor service {service} permits only locked read-only image mounts"
        )
    source = str(raw.get("source") or "")
    target = str(raw.get("target") or "")
    subpath = Path(str(image.get("subpath") or ""))
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", source):
        raise ValueError(
            f"Harbor service {service} image mount must use an exact image ID"
        )
    if (
        not target
        or not Path(target).is_absolute()
        or ".." in Path(target).parts
        or subpath.is_absolute()
        or not subpath.parts
        or ".." in subpath.parts
    ):
        raise ValueError(f"Harbor service {service} has an invalid image mount path")


def _locked_path(raw: Any, repo_root: Path) -> Path:
    path = Path(str(raw))
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        raise ValueError("Harbor Compose fragment escapes the Fugue checkout")
    return resolved


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [
            item
            for nested in value.values()
            for item in _string_values(nested)
        ]
    if isinstance(value, Sequence):
        return [item for nested in value for item in _string_values(nested)]
    return []
