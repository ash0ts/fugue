from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import re
import shlex
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.connectivity_gateway import mint_cell_capability

COREWEAVE_LOCK_SCHEMA_VERSION = 1
COREWEAVE_ATTESTATION_SCHEMA_VERSION = 1
COREWEAVE_FAILURE_SCHEMA_VERSION = 1
COREWEAVE_LOCK_PATH = Path(".fugue/coreweave-profile.lock.json")
COREWEAVE_HARBOR_VERSION = "0.18.0"
COREWEAVE_SDK_VERSION = "0.26.0"
COREWEAVE_ENVIRONMENT_IMPORT = (
    "fugue.bench.coreweave:FugueCoreWeaveEnvironment"
)
COREWEAVE_ATTESTATION_NAME = "coreweave-sandbox-attestation.json"
COREWEAVE_OPERATION_NAME = "coreweave-operation.json"
COREWEAVE_FAILURE_NAME = "coreweave-failure.json"
COREWEAVE_CAPABILITY_TOKEN_ENV = "FUGUE_CONNECTIVITY_TOKEN"
COREWEAVE_GATEWAY_CA_PATH = "/etc/fugue/gateway-ca.pem"
COREWEAVE_RUNTIME_MANIFEST_PATH = "/etc/fugue/runtime-manifest.json"
_DIGEST_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_SENSITIVE_NAME = re.compile(
    r"(?:^|_)(?:api_?key|token|secret|password|credential|private_?key)(?:$|_)",
    re.IGNORECASE,
)
_MAX_ARTIFACT_FILES = 10_000
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
_MAX_TRANSFER_ARCHIVE_BYTES = 256 * 1024 * 1024


def validate_coreweave_artifact_archive(
    archive_path: Path,
) -> tuple[tarfile.TarInfo, ...]:
    """Reject unsafe or unbounded sandbox output before host extraction."""
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = tuple(archive.getmembers())
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError("CoreWeave artifact archive is malformed") from exc
    if len(members) > _MAX_ARTIFACT_FILES:
        raise RuntimeError("CoreWeave artifact archive contains too many entries")
    total_bytes = 0
    for member in members:
        name = PurePosixPath(member.name)
        if (
            name.is_absolute()
            or ".." in name.parts
            or "\\" in member.name
            or "\x00" in member.name
        ):
            raise RuntimeError(
                "CoreWeave artifact archive contains path traversal"
            )
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(
                "CoreWeave artifact archive contains a non-regular entry"
            )
        if member.isfile():
            if member.size < 0:
                raise RuntimeError(
                    "CoreWeave artifact archive contains an invalid size"
                )
            total_bytes += member.size
            if total_bytes > _MAX_ARTIFACT_BYTES:
                raise RuntimeError(
                    "CoreWeave artifact archive exceeds the expanded byte limit"
                )
    return members


@dataclass(frozen=True)
class CoreWeaveResourcesV1:
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CoreWeaveNetworkLockV1:
    selected_mode: Literal["none", "gateway"]
    egress_mode: str
    ingress_mode: None = None
    gateway_cidrs: tuple[str, ...] = ()
    gateway_hosts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoreWeaveGatewayLockV1:
    base_url: str
    policy_sha256: str
    route_ids: tuple[str, ...]
    certificate_sha256: str
    max_requests: int = 128
    max_request_bytes: int = 8 * 1024 * 1024
    max_response_bytes: int = 64 * 1024 * 1024

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoreWeaveRuntimeAssetV1:
    kind: Literal["fugue", "agent", "skill", "mcp", "context", "task"]
    id: str
    digest: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CoreWeaveRuntimeManifestV1:
    schema_version: int
    python_version: str
    assets: tuple[CoreWeaveRuntimeAssetV1, ...]
    manifest_sha256: str = ""

    def unsigned_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["manifest_sha256"] = ""
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoreWeaveSandboxLockV1:
    schema_version: int
    harbor_version: str
    cwsandbox_version: str
    runner_id: str
    profile_id: str
    profile_name: str
    profile_document_sha256: str
    image: str
    runtime_class: str
    namespace_strategy: Literal["per-user"]
    network: CoreWeaveNetworkLockV1
    resources: CoreWeaveResourcesV1
    max_lifetime_seconds: int
    runtime_manifest: CoreWeaveRuntimeManifestV1
    gateway: CoreWeaveGatewayLockV1 | None = None
    approved_secret_bindings: tuple[str, ...] = ()
    lock_sha256: str = ""

    def unsigned_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["lock_sha256"] = ""
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectiveSandboxAttestationV1:
    schema_version: int
    sandbox_id: str
    runner_id: str
    profile_id: str
    applied_egress_mode: str
    applied_ingress_mode: str | None
    resource_requests: dict[str, str]
    resource_limits: dict[str, str]
    image: str
    runtime_class: str
    profile_document_sha256: str
    runtime_manifest_sha256: str
    started_at: str
    stopped_at: str | None
    deleted: bool
    lock_sha256: str
    eligible: bool
    failures: tuple[str, ...] = ()
    attestation_sha256: str = ""

    def unsigned_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["attestation_sha256"] = ""
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoreWeaveFailureV1:
    schema_version: int
    category: Literal[
        "authentication",
        "profile_mismatch",
        "scheduling",
        "startup",
        "execution",
        "timeout",
        "artifact",
        "attestation",
        "gateway",
        "teardown",
    ]
    operation: str
    retryable_before_agent_start: bool
    detail: str
    sandbox_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def effective_attestation_from_dict(
    raw: Mapping[str, Any],
) -> EffectiveSandboxAttestationV1:
    _reject_unknown(
        raw,
        {
            "schema_version",
            "sandbox_id",
            "runner_id",
            "profile_id",
            "applied_egress_mode",
            "applied_ingress_mode",
            "resource_requests",
            "resource_limits",
            "image",
            "runtime_class",
            "profile_document_sha256",
            "runtime_manifest_sha256",
            "started_at",
            "stopped_at",
            "deleted",
            "lock_sha256",
            "eligible",
            "failures",
            "attestation_sha256",
        },
        "CoreWeave sandbox attestation",
    )
    if int(raw.get("schema_version") or 0) != COREWEAVE_ATTESTATION_SCHEMA_VERSION:
        raise ValueError("unsupported CoreWeave sandbox attestation schema")
    failures = _text_tuple(raw.get("failures") or ())
    eligible = raw.get("eligible")
    deleted = raw.get("deleted")
    if not isinstance(eligible, bool) or not isinstance(deleted, bool):
        raise ValueError("CoreWeave attestation eligibility fields must be boolean")
    if eligible and failures:
        raise ValueError("eligible CoreWeave attestation may not contain failures")
    attestation = EffectiveSandboxAttestationV1(
        schema_version=COREWEAVE_ATTESTATION_SCHEMA_VERSION,
        sandbox_id=_nonempty(raw.get("sandbox_id"), "sandbox id"),
        runner_id=_identifier(raw.get("runner_id"), "runner id"),
        profile_id=_identifier(raw.get("profile_id"), "profile id"),
        applied_egress_mode=_identifier(
            raw.get("applied_egress_mode"), "applied egress mode"
        ),
        applied_ingress_mode=(
            str(raw["applied_ingress_mode"])
            if raw.get("applied_ingress_mode") is not None
            else None
        ),
        resource_requests=_string_mapping(
            raw.get("resource_requests"), "resource requests"
        ),
        resource_limits=_string_mapping(
            raw.get("resource_limits"), "resource limits"
        ),
        image=_digest_image(raw.get("image")),
        runtime_class=_nonempty(raw.get("runtime_class"), "runtime class"),
        profile_document_sha256=_sha256(
            raw.get("profile_document_sha256"), "profile document digest"
        ),
        runtime_manifest_sha256=_sha256(
            raw.get("runtime_manifest_sha256"), "runtime manifest digest"
        ),
        started_at=_nonempty(raw.get("started_at"), "sandbox start time"),
        stopped_at=(
            _nonempty(raw.get("stopped_at"), "sandbox stop time")
            if raw.get("stopped_at") is not None
            else None
        ),
        deleted=deleted,
        lock_sha256=_sha256(raw.get("lock_sha256"), "sandbox lock digest"),
        eligible=eligible,
        failures=failures,
    )
    supplied = _sha256(raw.get("attestation_sha256"), "attestation digest")
    digest = stable_digest(attestation.unsigned_dict())
    if not hmac.compare_digest(supplied, digest):
        raise ValueError("CoreWeave attestation digest does not match")
    return replace(attestation, attestation_sha256=digest)


def runtime_manifest_from_dict(
    raw: Mapping[str, Any],
) -> CoreWeaveRuntimeManifestV1:
    _reject_unknown(
        raw,
        {"schema_version", "python_version", "assets", "manifest_sha256"},
        "CoreWeave runtime manifest",
    )
    if int(raw.get("schema_version") or 0) != 1:
        raise ValueError("unsupported CoreWeave runtime manifest schema")
    python_version = _nonempty(
        raw.get("python_version"), "CoreWeave runtime Python version"
    )
    if python_version != "3.13":
        raise ValueError("CoreWeave runtime must pin Python 3.13")
    assets_raw = raw.get("assets")
    if not isinstance(assets_raw, Sequence) or isinstance(assets_raw, (str, bytes)):
        raise ValueError("CoreWeave runtime assets must be a list")
    assets: list[CoreWeaveRuntimeAssetV1] = []
    identities: set[tuple[str, str]] = set()
    paths: set[str] = set()
    for index, value in enumerate(assets_raw):
        asset_raw = _mapping(value, f"CoreWeave runtime asset {index}")
        _reject_unknown(
            asset_raw,
            {"kind", "id", "digest", "path"},
            f"CoreWeave runtime asset {index}",
        )
        kind = str(asset_raw.get("kind") or "")
        if kind not in {"fugue", "agent", "skill", "mcp", "context", "task"}:
            raise ValueError(f"unsupported CoreWeave runtime asset kind: {kind!r}")
        asset_id = _identifier(asset_raw.get("id"), "runtime asset id")
        digest_value = str(asset_raw.get("digest") or "")
        if digest_value.startswith("sha256:"):
            digest_value = digest_value.removeprefix("sha256:")
        digest = _sha256(digest_value, "runtime asset digest")
        path = _safe_runtime_path(asset_raw.get("path"))
        identity = (kind, asset_id)
        if identity in identities:
            raise ValueError(
                f"duplicate CoreWeave runtime asset: {kind}:{asset_id}"
            )
        if path in paths:
            raise ValueError(f"duplicate CoreWeave runtime asset path: {path}")
        identities.add(identity)
        paths.add(path)
        assets.append(
            CoreWeaveRuntimeAssetV1(
                kind=kind,  # type: ignore[arg-type]
                id=asset_id,
                digest=digest,
                path=path,
            )
        )
    if not assets:
        raise ValueError("CoreWeave runtime manifest must contain assets")
    manifest = CoreWeaveRuntimeManifestV1(
        schema_version=1,
        python_version=python_version,
        assets=tuple(sorted(assets, key=lambda item: (item.kind, item.id))),
    )
    digest = stable_digest(manifest.unsigned_dict())
    supplied = str(raw.get("manifest_sha256") or "")
    if supplied and not hmac.compare_digest(supplied, digest):
        raise ValueError("CoreWeave runtime manifest digest does not match")
    return replace(manifest, manifest_sha256=digest)


def read_runtime_manifest(path: Path) -> CoreWeaveRuntimeManifestV1:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read CoreWeave runtime manifest {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("CoreWeave runtime manifest must be an object")
    return runtime_manifest_from_dict(raw)


def coreweave_lock_from_dict(  # noqa: C901
    raw: Mapping[str, Any],
) -> CoreWeaveSandboxLockV1:
    _reject_unknown(
        raw,
        {
            "schema_version",
            "harbor_version",
            "cwsandbox_version",
            "runner_id",
            "profile_id",
            "profile_name",
            "profile_document_sha256",
            "image",
            "runtime_class",
            "namespace_strategy",
            "network",
            "resources",
            "max_lifetime_seconds",
            "runtime_manifest",
            "gateway",
            "approved_secret_bindings",
            "lock_sha256",
        },
        "CoreWeave sandbox lock",
    )
    if int(raw.get("schema_version") or 0) != COREWEAVE_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported CoreWeave sandbox lock schema")
    network_raw = _mapping(raw.get("network"), "CoreWeave network lock")
    _reject_unknown(
        network_raw,
        {
            "selected_mode",
            "egress_mode",
            "ingress_mode",
            "gateway_cidrs",
            "gateway_hosts",
        },
        "CoreWeave network lock",
    )
    mode = str(network_raw.get("selected_mode") or "")
    if mode not in {"none", "gateway"}:
        raise ValueError("CoreWeave network mode must be none or gateway")
    if network_raw.get("ingress_mode") is not None:
        raise ValueError("CoreWeave sandbox ingress is forbidden")
    cidrs = _text_tuple(network_raw.get("gateway_cidrs") or ())
    for cidr in cidrs:
        parsed = ipaddress.ip_network(cidr, strict=True)
        minimum_prefix = 24 if parsed.version == 4 else 64
        if parsed.prefixlen < minimum_prefix:
            raise ValueError(
                f"CoreWeave gateway CIDR is too broad: {cidr}; "
                f"require /{minimum_prefix} or narrower"
            )
    hosts = _text_tuple(network_raw.get("gateway_hosts") or ())
    network = CoreWeaveNetworkLockV1(
        selected_mode=mode,  # type: ignore[arg-type]
        egress_mode=_nonempty(network_raw.get("egress_mode"), "egress mode"),
        gateway_cidrs=cidrs,
        gateway_hosts=hosts,
    )
    expected_egress_mode = "deny-all" if mode == "none" else "fugue-gateway"
    if network.egress_mode != expected_egress_mode:
        raise ValueError(
            f"CoreWeave egress mode must be {expected_egress_mode}"
        )
    resources_raw = _mapping(raw.get("resources"), "CoreWeave resources")
    _reject_unknown(
        resources_raw,
        {"cpu_request", "cpu_limit", "memory_request", "memory_limit"},
        "CoreWeave resources",
    )
    resources = CoreWeaveResourcesV1(
        cpu_request=_nonempty(resources_raw.get("cpu_request"), "CPU request"),
        cpu_limit=_nonempty(resources_raw.get("cpu_limit"), "CPU limit"),
        memory_request=_nonempty(
            resources_raw.get("memory_request"), "memory request"
        ),
        memory_limit=_nonempty(resources_raw.get("memory_limit"), "memory limit"),
    )
    runtime_manifest = runtime_manifest_from_dict(
        _mapping(raw.get("runtime_manifest"), "CoreWeave runtime manifest")
    )
    gateway_raw = raw.get("gateway")
    gateway = None
    if gateway_raw is not None:
        selected = _mapping(gateway_raw, "CoreWeave gateway lock")
        _reject_unknown(
            selected,
            {
                "base_url",
                "policy_sha256",
                "route_ids",
                "certificate_sha256",
                "max_requests",
                "max_request_bytes",
                "max_response_bytes",
            },
            "CoreWeave gateway lock",
        )
        gateway = CoreWeaveGatewayLockV1(
            base_url=_https_url(selected.get("base_url")),
            policy_sha256=_sha256(selected.get("policy_sha256"), "gateway policy"),
            route_ids=_text_tuple(selected.get("route_ids") or ()),
            certificate_sha256=_sha256(
                selected.get("certificate_sha256"), "gateway certificate"
            ),
            max_requests=_positive_int(
                selected.get("max_requests", 128), "gateway request limit"
            ),
            max_request_bytes=_positive_int(
                selected.get("max_request_bytes", 8 * 1024 * 1024),
                "gateway request size",
            ),
            max_response_bytes=_positive_int(
                selected.get("max_response_bytes", 64 * 1024 * 1024),
                "gateway response size",
            ),
        )
    lifetime = _positive_int(
        raw.get("max_lifetime_seconds"), "CoreWeave maximum lifetime"
    )
    if lifetime > 3600:
        raise ValueError("CoreWeave maximum lifetime may not exceed 3600 seconds")
    image = _nonempty(raw.get("image"), "CoreWeave image")
    if not _DIGEST_IMAGE.fullmatch(image):
        raise ValueError("CoreWeave image must use an exact sha256 registry digest")
    runtime_class = _nonempty(raw.get("runtime_class"), "runtime class")
    if runtime_class != "kata-qemu":
        raise ValueError("CoreWeave runtime_class must be kata-qemu")
    namespace = str(raw.get("namespace_strategy") or "")
    if namespace != "per-user":
        raise ValueError("CoreWeave namespace strategy must be per-user")
    if mode == "gateway":
        if gateway is None or not cidrs or not hosts:
            raise ValueError(
                "gateway mode requires a gateway lock, CIDRs, and hostnames"
            )
        if network.egress_mode == "internet":
            raise ValueError("unrestricted CoreWeave internet egress is forbidden")
        from urllib.parse import urlparse

        gateway_host = str(urlparse(gateway.base_url).hostname or "")
        if len(hosts) != 1 or hosts[0] != gateway_host:
            raise ValueError(
                "CoreWeave gateway hostname must exactly match its locked base URL"
            )
    elif gateway is not None or cidrs or hosts:
        raise ValueError("none network mode may not carry gateway configuration")
    lock = CoreWeaveSandboxLockV1(
        schema_version=COREWEAVE_LOCK_SCHEMA_VERSION,
        harbor_version=_nonempty(raw.get("harbor_version"), "Harbor version"),
        cwsandbox_version=_nonempty(
            raw.get("cwsandbox_version"), "cwsandbox version"
        ),
        runner_id=_identifier(raw.get("runner_id"), "runner id"),
        profile_id=_identifier(raw.get("profile_id"), "profile id"),
        profile_name=_identifier(raw.get("profile_name"), "profile name"),
        profile_document_sha256=_sha256(
            raw.get("profile_document_sha256"), "profile document"
        ),
        image=image,
        runtime_class=runtime_class,
        namespace_strategy=namespace,  # type: ignore[arg-type]
        network=network,
        resources=resources,
        max_lifetime_seconds=lifetime,
        runtime_manifest=runtime_manifest,
        gateway=gateway,
        approved_secret_bindings=_text_tuple(
            raw.get("approved_secret_bindings") or ()
        ),
    )
    if lock.harbor_version != COREWEAVE_HARBOR_VERSION:
        raise ValueError(
            f"CoreWeave lock must pin harbor {COREWEAVE_HARBOR_VERSION}"
        )
    if lock.cwsandbox_version != COREWEAVE_SDK_VERSION:
        raise ValueError(
            f"CoreWeave lock must pin cwsandbox {COREWEAVE_SDK_VERSION}"
        )
    if any(_SENSITIVE_NAME.search(item) for item in lock.approved_secret_bindings):
        raise ValueError(
            "CoreWeave lock contains secret-like values; use logical binding ids"
        )
    expected = stable_digest(lock.unsigned_dict())
    supplied = str(raw.get("lock_sha256") or "")
    if supplied and supplied != expected:
        raise ValueError("CoreWeave sandbox lock digest does not match")
    return replace(lock, lock_sha256=expected)


def read_coreweave_lock(path: Path) -> CoreWeaveSandboxLockV1:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read CoreWeave sandbox lock {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("CoreWeave sandbox lock must be an object")
    return coreweave_lock_from_dict(raw)


def write_coreweave_lock(
    path: Path, lock: CoreWeaveSandboxLockV1
) -> CoreWeaveSandboxLockV1:
    normalized = coreweave_lock_from_dict(lock.to_dict())
    atomic_write_json(path, normalized.to_dict())
    path.chmod(0o600)
    return normalized


def build_coreweave_lock(
    *,
    runner_id: str,
    profile_id: str,
    profile_name: str,
    profile_document: bytes,
    image: str,
    runtime_manifest: CoreWeaveRuntimeManifestV1,
    network: Literal["none", "gateway"],
    gateway: CoreWeaveGatewayLockV1 | None,
    gateway_cidrs: Sequence[str] = (),
    gateway_hosts: Sequence[str] = (),
    max_lifetime_seconds: int = 1800,
) -> CoreWeaveSandboxLockV1:
    validate_coreweave_profile_document(
        profile_document,
        profile_name=profile_name,
        image=image,
        network=network,
        gateway_cidrs=gateway_cidrs,
    )
    raw = {
        "schema_version": COREWEAVE_LOCK_SCHEMA_VERSION,
        "harbor_version": COREWEAVE_HARBOR_VERSION,
        "cwsandbox_version": COREWEAVE_SDK_VERSION,
        "runner_id": runner_id,
        "profile_id": profile_id,
        "profile_name": profile_name,
        "profile_document_sha256": hashlib.sha256(profile_document).hexdigest(),
        "image": image,
        "runtime_class": "kata-qemu",
        "namespace_strategy": "per-user",
        "network": {
            "selected_mode": network,
            "egress_mode": "deny-all" if network == "none" else "fugue-gateway",
            "ingress_mode": None,
            "gateway_cidrs": list(gateway_cidrs),
            "gateway_hosts": list(gateway_hosts),
        },
        "resources": {
            "cpu_request": "500m",
            "cpu_limit": "2",
            "memory_request": "512Mi",
            "memory_limit": "4Gi",
        },
        "max_lifetime_seconds": max_lifetime_seconds,
        "runtime_manifest": runtime_manifest.to_dict(),
        "gateway": gateway.to_dict() if gateway is not None else None,
        "approved_secret_bindings": [],
    }
    return coreweave_lock_from_dict(raw)


def validate_coreweave_profile_document(  # noqa: C901
    profile_document: bytes,
    *,
    profile_name: str,
    image: str,
    network: Literal["none", "gateway"],
    gateway_cidrs: Sequence[str] = (),
) -> dict[str, Any]:
    """Fail closed on the administrator-owned fields Fugue relies on."""
    try:
        raw = yaml.safe_load(profile_document)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid CoreWeave profile YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("CoreWeave profile must be an object")
    profile = {str(key): value for key, value in raw.items()}
    _reject_unknown(profile, {"display_name", "description", "spec"}, "profile")
    if str(profile.get("display_name") or "") != profile_name:
        raise ValueError("CoreWeave profile display name differs from the lock")
    spec_raw = profile.get("spec")
    if not isinstance(spec_raw, Mapping):
        raise ValueError("CoreWeave profile spec must be an object")
    spec = {str(key): value for key, value in spec_raw.items()}
    _reject_unknown(
        spec,
        {
            "container_image",
            "runtime_class",
            "resource_defaults",
            "namespace",
            "network",
            "pod",
        },
        "CoreWeave profile spec",
    )
    if str(spec.get("container_image") or "") != image:
        raise ValueError("CoreWeave profile image differs from the digest-pinned lock")
    if str(spec.get("runtime_class") or "") != "kata-qemu":
        raise ValueError("CoreWeave profile must require runtime_class kata-qemu")
    namespace = spec.get("namespace")
    if not isinstance(namespace, Mapping) or namespace.get("strategy") != "per-user":
        raise ValueError("CoreWeave profile must use per-user namespaces")
    _reject_unknown(
        namespace,
        {"strategy", "namespacePrefix", "autoCreate"},
        "CoreWeave namespace",
    )
    if (
        namespace.get("namespacePrefix") != "fugue-"
        or namespace.get("autoCreate") is not True
    ):
        raise ValueError("CoreWeave namespace settings differ from policy")
    network_raw = spec.get("network")
    if not isinstance(network_raw, Mapping):
        raise ValueError("CoreWeave profile must declare an explicit network policy")
    _reject_unknown(network_raw, {"ingress", "egress"}, "CoreWeave network")
    if network_raw.get("ingress"):
        raise ValueError("CoreWeave profile may not define ingress")
    egress = network_raw.get("egress")
    if not isinstance(egress, Mapping):
        raise ValueError("CoreWeave profile must declare egress modes")
    _reject_unknown(egress, {"default", "modes"}, "CoreWeave egress")
    modes = egress.get("modes")
    if not isinstance(modes, Mapping):
        raise ValueError("CoreWeave profile egress modes must be an object")
    expected_mode = "deny-all" if network == "none" else "fugue-gateway"
    if set(modes) != {"deny-all", "fugue-gateway"}:
        raise ValueError("CoreWeave profile contains an unapproved egress mode")
    if str(egress.get("default") or "") != "deny-all":
        raise ValueError("CoreWeave profile default egress must be deny-all")
    deny_all = modes.get("deny-all")
    if not isinstance(deny_all, Mapping) or deny_all.get("type") != "none":
        raise ValueError("CoreWeave profile must include deny-all egress")
    if set(deny_all) != {"type"}:
        raise ValueError("CoreWeave deny-all mode contains unsupported fields")
    gateway_mode = modes.get("fugue-gateway")
    if (
        not isinstance(gateway_mode, Mapping)
        or gateway_mode.get("type") != "allowlist"
        or set(gateway_mode) != {"type", "cidrs"}
    ):
        raise ValueError("CoreWeave profile gateway mode must be an exact allowlist")
    profile_gateway_cidrs = tuple(
        str(value) for value in gateway_mode.get("cidrs") or ()
    )
    if not profile_gateway_cidrs:
        raise ValueError("CoreWeave profile gateway allowlist must not be empty")
    for cidr in profile_gateway_cidrs:
        parsed = ipaddress.ip_network(cidr, strict=True)
        minimum_prefix = 24 if parsed.version == 4 else 64
        if parsed.prefixlen < minimum_prefix:
            raise ValueError("CoreWeave profile gateway CIDR is too broad")
    selected_mode = modes.get(expected_mode)
    if not isinstance(selected_mode, Mapping):
        raise ValueError(f"CoreWeave profile is missing egress mode {expected_mode}")
    expected_type = "none" if network == "none" else "allowlist"
    if selected_mode.get("type") != expected_type:
        raise ValueError("CoreWeave profile egress type differs from the lock")
    observed_cidrs = tuple(str(value) for value in selected_mode.get("cidrs") or ())
    if tuple(gateway_cidrs) != observed_cidrs:
        raise ValueError("CoreWeave profile gateway CIDRs differ from the lock")
    resources = spec.get("resource_defaults")
    if resources != {
        "cpuRequest": "500m",
        "cpuLimit": "2",
        "memoryRequest": "512Mi",
        "memoryLimit": "4Gi",
    }:
        raise ValueError("CoreWeave profile resources differ from the Fugue policy")
    pod = spec.get("pod")
    if not isinstance(pod, Mapping) or not isinstance(pod.get("spec"), Mapping):
        raise ValueError("CoreWeave profile must declare pod security controls")
    _reject_unknown(pod, {"spec"}, "CoreWeave pod")
    pod_spec = pod["spec"]
    allowed_pod_fields = {
        "automountServiceAccountToken",
        "hostIPC",
        "hostNetwork",
        "hostPID",
        "securityContext",
        "containers",
        "volumes",
    }
    unknown_pod_fields = sorted(set(pod_spec) - allowed_pod_fields)
    if unknown_pod_fields:
        raise ValueError(
            "CoreWeave profile contains unapproved pod fields: "
            + ", ".join(unknown_pod_fields)
        )
    security_context = pod_spec.get("securityContext")
    if not isinstance(security_context, Mapping):
        raise ValueError("CoreWeave profile must declare a pod security context")
    if security_context.get("runAsNonRoot") is not True:
        raise ValueError("CoreWeave profile must require a non-root process")
    if security_context.get("runAsUser") in {None, 0, "0"}:
        raise ValueError("CoreWeave profile must pin a non-root uid")
    if pod_spec.get("automountServiceAccountToken") is not False:
        raise ValueError("CoreWeave profile must disable service-account tokens")
    if any(pod_spec.get(key) is True for key in ("hostIPC", "hostNetwork", "hostPID")):
        raise ValueError("CoreWeave profile may not share host namespaces")
    containers = pod_spec.get("containers")
    if not isinstance(containers, list) or len(containers) != 1:
        raise ValueError("CoreWeave profile must constrain the single main container")
    container = containers[0]
    allowed_container_fields = {"name", "securityContext", "volumeMounts"}
    unknown_container_fields = sorted(set(container) - allowed_container_fields)
    if unknown_container_fields:
        raise ValueError(
            "CoreWeave main container contains unsupported fields: "
            + ", ".join(unknown_container_fields)
        )
    if not isinstance(container, Mapping):
        raise ValueError("CoreWeave container policy must be an object")
    if set(container) != {"name", "securityContext", "volumeMounts"}:
        raise ValueError("CoreWeave main container contains unapproved fields")
    if container.get("name") != "main":
        raise ValueError("CoreWeave profile must constrain the main container")
    container_security = container.get("securityContext")
    if not isinstance(container_security, Mapping):
        raise ValueError("CoreWeave profile must declare container security controls")
    required = {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    if any(container_security.get(key) != value for key, value in required.items()):
        raise ValueError("CoreWeave container security controls differ from policy")
    expected_mounts = {
        ("workspace", "/workspace"),
        ("tmp", "/tmp"),
        ("artifacts", "/logs"),
        ("tests", "/tests"),
        ("solution", "/solution"),
        ("harbor", "/harbor"),
    }
    observed_mounts = {
        (str(value.get("name") or ""), str(value.get("mountPath") or ""))
        for value in container.get("volumeMounts") or ()
        if isinstance(value, Mapping)
    }
    if observed_mounts != expected_mounts:
        raise ValueError("CoreWeave writable paths differ from policy")
    volume_names = {
        str(value.get("name") or "")
        for value in pod_spec.get("volumes") or ()
        if isinstance(value, Mapping)
    }
    if volume_names != {
        "workspace",
        "tmp",
        "artifacts",
        "tests",
        "solution",
        "harbor",
    }:
        raise ValueError("CoreWeave profile must expose only bounded emptyDir volumes")
    if any(
        set(value) - {"name", "emptyDir"}
        for value in pod_spec.get("volumes") or ()
        if isinstance(value, Mapping)
    ):
        raise ValueError("CoreWeave profile contains an unapproved volume type")
    if any(
        not isinstance(value.get("emptyDir"), Mapping)
        or set(value["emptyDir"]) != {"sizeLimit"}
        for value in pod_spec.get("volumes") or ()
        if isinstance(value, Mapping)
    ):
        raise ValueError("CoreWeave emptyDir volumes must have exact size limits")
    return profile


def coreweave_execution_identity(
    environment: Mapping[str, Any],
) -> dict[str, Any] | None:
    if environment.get("import_path") != COREWEAVE_ENVIRONMENT_IMPORT:
        return None
    kwargs = _mapping(environment.get("kwargs"), "CoreWeave environment kwargs")
    lock = coreweave_lock_from_dict(
        _mapping(kwargs.get("fugue_coreweave_lock"), "CoreWeave sandbox lock")
    )
    return {
        "backend": "coreweave",
        "lock": lock.to_dict(),
        "lock_sha256": lock.lock_sha256,
        "runner_id": lock.runner_id,
        "profile_id": lock.profile_id,
        "profile_name": lock.profile_name,
        "image": lock.image,
        "runtime_manifest_sha256": lock.runtime_manifest.manifest_sha256,
        "network": lock.network.to_dict(),
        "gateway_policy_sha256": (
            lock.gateway.policy_sha256 if lock.gateway is not None else None
        ),
    }


def require_coreweave_runtime_assets(
    lock: CoreWeaveSandboxLockV1,
    requirements: Sequence[Mapping[str, str]],
) -> None:
    available = {
        (asset.kind, asset.id): asset.digest for asset in lock.runtime_manifest.assets
    }
    missing: list[str] = []
    drifted: list[str] = []
    for requirement in requirements:
        kind = str(requirement.get("kind") or "")
        asset_id = str(requirement.get("id") or "")
        digest = str(requirement.get("digest") or "").removeprefix("sha256:")
        observed = available.get((kind, asset_id))
        if observed is None:
            missing.append(f"{kind}:{asset_id}")
        elif not hmac.compare_digest(observed, digest):
            drifted.append(f"{kind}:{asset_id}")
    if missing:
        raise ValueError(
            "CoreWeave runtime image is missing locked assets: "
            + ", ".join(sorted(missing))
        )
    if drifted:
        raise ValueError(
            "CoreWeave runtime image asset digests differ from the comparison: "
            + ", ".join(sorted(drifted))
        )


def comparison_environment(
    lock: CoreWeaveSandboxLockV1,
    *,
    network: Literal["none", "gateway"],
    max_lifetime_seconds: int,
) -> dict[str, Any]:
    if network != lock.network.selected_mode:
        raise ValueError("comparison network differs from the CoreWeave profile lock")
    if max_lifetime_seconds != lock.max_lifetime_seconds:
        raise ValueError(
            "comparison maximum lifetime differs from the CoreWeave profile lock"
        )
    allowed_hosts = (
        list(lock.network.gateway_hosts) if network == "gateway" else None
    )
    return {
        "import_path": COREWEAVE_ENVIRONMENT_IMPORT,
        "force_build": False,
        "delete": True,
        "docker_image": lock.image,
        "network_mode": "allowlist" if network == "gateway" else "no-network",
        "allowed_hosts": allowed_hosts,
        "override_cpus": 2,
        "override_memory_mb": 4096,
        "kwargs": {
            "fugue_coreweave_lock": lock.to_dict(),
            "profile_ids": [lock.profile_id],
            "profile_names": [lock.profile_name],
            "runner_ids": [lock.runner_id],
            "max_lifetime_seconds": lock.max_lifetime_seconds,
            "egress_mode": lock.network.egress_mode,
        },
    }


def bind_coreweave_job(
    environment: Mapping[str, Any],
    *,
    instance_id: str,
    run_id: str,
    job_name: str,
    execution_fingerprint: str,
) -> dict[str, Any]:
    value = json.loads(json.dumps(environment))
    if value.get("import_path") != COREWEAVE_ENVIRONMENT_IMPORT:
        return value
    kwargs = _mapping(value.get("kwargs"), "CoreWeave environment kwargs")
    lock = coreweave_lock_from_dict(
        _mapping(kwargs.get("fugue_coreweave_lock"), "CoreWeave sandbox lock")
    )
    instance = _identifier(instance_id, "Fugue instance id")
    tags = (
        "fugue",
        f"fi-{stable_digest({'instance_id': instance})[:20]}",
        f"fr-{stable_digest({'run_id': run_id})[:20]}",
        f"fc-{stable_digest({'job': job_name})[:20]}",
        f"fx-{execution_fingerprint[:20]}",
    )
    kwargs.update(
        {
            "tags": list(tags),
            "fugue_instance_id": instance,
            "attestation_name": COREWEAVE_ATTESTATION_NAME,
        }
    )
    value["kwargs"] = kwargs
    value["env"] = {
        **dict(value.get("env") or {}),
        COREWEAVE_CAPABILITY_TOKEN_ENV: (
            f"${{{COREWEAVE_CAPABILITY_TOKEN_ENV}}}"
            if lock.network.selected_mode == "gateway"
            else ""
        ),
        "FUGUE_INSTANCE_ID": "${FUGUE_INSTANCE_ID}",
        "FUGUE_RUN_ID": "${FUGUE_RUN_ID}",
        "FUGUE_CELL_ID": "${FUGUE_CELL_ID}",
        "FUGUE_EXECUTION_FINGERPRINT": execution_fingerprint,
    }
    return value


def coreweave_cell_environment(
    *,
    config_path: Path,
    cell_id: str,
    run_id: str,
    execution_fingerprint: str,
    worker_env: Mapping[str, str],
) -> dict[str, str]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read CoreWeave cell config: {exc}") from exc
    if not isinstance(config, Mapping):
        raise RuntimeError("CoreWeave cell config must be an object")
    environment = _mapping(config.get("environment"), "Harbor environment")
    identity = coreweave_execution_identity(environment)
    if identity is None:
        return {}
    kwargs = _mapping(environment.get("kwargs"), "CoreWeave environment kwargs")
    lock = coreweave_lock_from_dict(
        _mapping(kwargs.get("fugue_coreweave_lock"), "CoreWeave sandbox lock")
    )
    instance_id = _identifier(
        worker_env.get("FUGUE_INSTANCE_ID") or "fugue-local",
        "Fugue instance id",
    )
    planned_instance = _identifier(
        kwargs.get("fugue_instance_id"), "planned Fugue instance id"
    )
    if instance_id != planned_instance:
        raise RuntimeError("Fugue worker instance differs from the admitted cell")
    overlay = {
        "FUGUE_INSTANCE_ID": instance_id,
        "FUGUE_RUN_ID": run_id,
        "FUGUE_CELL_ID": cell_id,
    }
    if lock.network.selected_mode == "none":
        return overlay
    if lock.gateway is None:
        raise RuntimeError(
            "CoreWeave gateway network mode is missing its gateway lock"
        )
    encoded_key = worker_env.get("FUGUE_GATEWAY_SIGNING_KEY", "")
    try:
        signing_key = base64.b64decode(encoded_key, validate=True)
    except ValueError as exc:
        raise RuntimeError("FUGUE_GATEWAY_SIGNING_KEY must be base64") from exc
    overlay[COREWEAVE_CAPABILITY_TOKEN_ENV] = mint_cell_capability(
        signing_key=signing_key,
        issuer=instance_id,
        cell_id=cell_id,
        execution_fingerprint=execution_fingerprint,
        route_ids=lock.gateway.route_ids,
        lifetime_seconds=lock.max_lifetime_seconds,
        max_requests=lock.gateway.max_requests,
        max_request_bytes=lock.gateway.max_request_bytes,
        max_response_bytes=lock.gateway.max_response_bytes,
    )
    return overlay


def coreweave_gateway_environment(
    environment: Mapping[str, Any],
) -> dict[str, str]:
    identity = coreweave_execution_identity(environment)
    if identity is None:
        return {}
    kwargs = _mapping(environment.get("kwargs"), "CoreWeave environment kwargs")
    lock = coreweave_lock_from_dict(
        _mapping(kwargs.get("fugue_coreweave_lock"), "CoreWeave sandbox lock")
    )
    if lock.network.selected_mode == "none":
        return {}
    if lock.gateway is None:
        raise ValueError(
            "CoreWeave gateway network mode is missing its gateway lock"
        )
    missing = {"model", "wandb-api", "weave"} - set(
        lock.gateway.route_ids
    )
    if missing:
        raise ValueError(
            "CoreWeave gateway policy is missing required routes: "
            + ", ".join(sorted(missing))
        )
    base = lock.gateway.base_url.rstrip("/")
    token = f"${{{COREWEAVE_CAPABILITY_TOKEN_ENV}}}"
    return {
        "FUGUE_MODEL_GATEWAY_BASE_URL": f"{base}/routes/model",
        "FUGUE_BRIDGE_BASE_URL": f"{base}/routes/model",
        "FUGUE_WEAVE_BASE_URL": f"{base}/routes/wandb-api",
        "FUGUE_WEAVE_TRACE_SERVER_URL": f"{base}/routes/weave",
        "FUGUE_WEAVE_API_KEY": token,
        "WANDB_API_KEY": token,
        "OPENAI_API_KEY": token,
        "ANTHROPIC_API_KEY": token,
        "OPENAI_COMPATIBLE_API_KEY": token,
        "LITELLM_MASTER_KEY": token,
        "SSL_CERT_FILE": COREWEAVE_GATEWAY_CA_PATH,
        "REQUESTS_CA_BUNDLE": COREWEAVE_GATEWAY_CA_PATH,
        "CURL_CA_BUNDLE": COREWEAVE_GATEWAY_CA_PATH,
        "NODE_EXTRA_CA_CERTS": COREWEAVE_GATEWAY_CA_PATH,
    }


def coreweave_gateway_mcp_servers(
    environment: Mapping[str, Any],
    servers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    identity = coreweave_execution_identity(environment)
    if identity is None:
        return [dict(server) for server in servers]
    kwargs = _mapping(environment.get("kwargs"), "CoreWeave environment kwargs")
    lock = coreweave_lock_from_dict(
        _mapping(kwargs.get("fugue_coreweave_lock"), "CoreWeave sandbox lock")
    )
    rewritten: list[dict[str, Any]] = []
    for raw in servers:
        server = dict(raw)
        transport = str(
            server.get("transport")
            or ("stdio" if server.get("command") else "sse")
        )
        if transport == "stdio":
            rewritten.append(server)
            continue
        if lock.network.selected_mode != "gateway" or lock.gateway is None:
            raise ValueError(
                "remote MCP requires CoreWeave execution network: gateway"
            )
        integration_id = _identifier(
            server.get("integration_id"),
            "remote MCP integration id",
        )
        route_id = f"mcp-{integration_id}"
        if route_id not in lock.gateway.route_ids:
            raise ValueError(
                f"CoreWeave gateway policy is missing remote MCP route {route_id}"
            )
        original = urlsplit(str(server.get("url") or ""))
        if (
            original.scheme != "https"
            or not original.hostname
            or original.username
            or original.password
            or original.query
            or original.fragment
        ):
            raise ValueError(
                "CoreWeave remote MCP endpoints must be locked HTTPS URLs "
                "without credentials, query strings, or fragments"
            )
        path = original.path or "/"
        base = lock.gateway.base_url.rstrip("/")
        server["url"] = f"{base}/routes/{route_id}{path}"
        rewritten.append(server)
    return rewritten


def assert_coreweave_versions() -> None:
    for package, expected in (
        ("harbor", COREWEAVE_HARBOR_VERSION),
        ("cwsandbox", COREWEAVE_SDK_VERSION),
    ):
        try:
            observed = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"{package} is required for CoreWeave execution"
            ) from exc
        if observed != expected:
            raise RuntimeError(
                f"{package} version drift: expected {expected}, found {observed}"
            )


def validate_effective_attestation(
    lock: CoreWeaveSandboxLockV1,
    *,
    sandbox_id: str,
    runner_id: str,
    profile_id: str,
    applied_egress_mode: str,
    applied_ingress_mode: str | None,
    resource_requests: Mapping[str, Any],
    resource_limits: Mapping[str, Any],
    stopped_at: str | None = None,
    deleted: bool = False,
) -> EffectiveSandboxAttestationV1:
    failures: list[str] = []
    if runner_id != lock.runner_id:
        failures.append("runner differs from lock")
    if profile_id != lock.profile_id:
        failures.append("profile differs from lock")
    if applied_egress_mode != lock.network.egress_mode:
        failures.append("applied egress differs from lock")
    if applied_ingress_mode not in {None, "", "none"}:
        failures.append("sandbox unexpectedly exposes ingress")
    requests = {str(k): str(v) for k, v in resource_requests.items()}
    limits = {str(k): str(v) for k, v in resource_limits.items()}
    if requests != {
        "cpu": lock.resources.cpu_request,
        "memory": lock.resources.memory_request,
    }:
        failures.append("resource requests differ from lock")
    if limits != {
        "cpu": lock.resources.cpu_limit,
        "memory": lock.resources.memory_limit,
    }:
        failures.append("resource limits differ from lock")
    unsigned = EffectiveSandboxAttestationV1(
        schema_version=COREWEAVE_ATTESTATION_SCHEMA_VERSION,
        sandbox_id=sandbox_id,
        runner_id=runner_id,
        profile_id=profile_id,
        applied_egress_mode=applied_egress_mode,
        applied_ingress_mode=applied_ingress_mode,
        resource_requests=requests,
        resource_limits=limits,
        image=lock.image,
        runtime_class=lock.runtime_class,
        profile_document_sha256=lock.profile_document_sha256,
        runtime_manifest_sha256=lock.runtime_manifest.manifest_sha256,
        started_at=datetime.now(UTC).isoformat(),
        stopped_at=stopped_at,
        deleted=deleted,
        lock_sha256=lock.lock_sha256,
        eligible=not failures,
        failures=tuple(failures),
    )
    return replace(
        unsigned,
        attestation_sha256=stable_digest(unsigned.unsigned_dict()),
    )


def classify_coreweave_failure(exc: BaseException, operation: str) -> CoreWeaveFailureV1:
    name = type(exc).__name__.lower()
    detail = str(exc)
    lowered = f"{name} {detail}".lower()
    if operation in {"artifact", "attestation", "gateway"}:
        category = operation
    elif "auth" in lowered or "permission" in lowered:
        category = "authentication"
    elif "profile" in lowered or "suitable_runner" in lowered:
        category = "profile_mismatch"
    elif "schedul" in lowered or "runner" in lowered:
        category = "scheduling"
    elif "timeout" in lowered:
        category = "timeout"
    elif operation in {"stop", "delete"}:
        category = "teardown"
    elif operation == "start":
        category = "startup"
    else:
        category = "execution"
    return CoreWeaveFailureV1(
        schema_version=COREWEAVE_FAILURE_SCHEMA_VERSION,
        category=category,  # type: ignore[arg-type]
        operation=operation,
        retryable_before_agent_start=category in {"scheduling", "startup", "timeout"},
        detail=_redacted_failure_detail(detail),
        sandbox_id=getattr(exc, "sandbox_id", None),
    )


def coreweave_failure_from_dict(
    raw: Mapping[str, Any],
) -> CoreWeaveFailureV1:
    _reject_unknown(
        raw,
        {
            "schema_version",
            "category",
            "operation",
            "retryable_before_agent_start",
            "detail",
            "sandbox_id",
        },
        "CoreWeave failure",
    )
    if raw.get("schema_version") != COREWEAVE_FAILURE_SCHEMA_VERSION:
        raise ValueError("unsupported CoreWeave failure schema version")
    category = _nonempty(raw.get("category"), "CoreWeave failure category")
    allowed_categories = {
        "authentication",
        "profile_mismatch",
        "scheduling",
        "startup",
        "execution",
        "timeout",
        "artifact",
        "attestation",
        "gateway",
        "teardown",
    }
    if category not in allowed_categories:
        raise ValueError("invalid CoreWeave failure category")
    operation = _identifier(
        raw.get("operation"),
        "CoreWeave failure operation",
    )
    retryable = raw.get("retryable_before_agent_start")
    if not isinstance(retryable, bool):
        raise ValueError(
            "CoreWeave failure retryable_before_agent_start must be a boolean"
        )
    detail = _nonempty(raw.get("detail"), "CoreWeave failure detail")
    sandbox_raw = raw.get("sandbox_id")
    sandbox_id = (
        None
        if sandbox_raw is None
        else _identifier(sandbox_raw, "CoreWeave failure sandbox id")
    )
    return CoreWeaveFailureV1(
        schema_version=COREWEAVE_FAILURE_SCHEMA_VERSION,
        category=category,  # type: ignore[arg-type]
        operation=operation,
        retryable_before_agent_start=retryable,
        detail=_redacted_failure_detail(detail),
        sandbox_id=sandbox_id,
    )


def _redacted_failure_detail(value: str) -> str:
    selected = value[:4000]
    selected = re.sub(
        r"(?i)\b(api[_-]?key|token|secret|password|credential)"
        r"\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        selected,
    )
    selected = re.sub(
        r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+",
        r"\1 [redacted]",
        selected,
    )
    return selected[:1000]


def coreweave_doctor(lock: CoreWeaveSandboxLockV1) -> dict[str, Any]:
    """Launch one disposable probe and return a redacted security report."""
    assert_coreweave_versions()
    if not _HARBOR_AVAILABLE:
        raise RuntimeError("CoreWeave execution requires the fugue[coreweave] extra")
    if not str(__import__("os").environ.get("CWSANDBOX_API_KEY") or ""):
        raise RuntimeError(
            "CWSANDBOX_API_KEY is required and must carry only SANDBOX_USER"
    )
    tags = ["fugue", "fugue-doctor", f"fl-{lock.lock_sha256[:20]}"]
    sandbox = None
    try:
        sandbox = __import__("cwsandbox").Sandbox.run(
            container_image=lock.image,
            profile_ids=[lock.profile_id],
            profile_names=[lock.profile_name],
            runner_ids=[lock.runner_id],
            network=__import__("cwsandbox").NetworkOptions(
                egress_mode=lock.network.egress_mode,
                ingress_mode=None,
            ),
            resources={
                "requests": {
                    "cpu": lock.resources.cpu_request,
                    "memory": lock.resources.memory_request,
                },
                "limits": {
                    "cpu": lock.resources.cpu_limit,
                    "memory": lock.resources.memory_limit,
                },
            },
            max_lifetime_seconds=min(lock.max_lifetime_seconds, 300),
            tags=tags,
        ).wait()
        identity = validate_effective_attestation(
            lock,
            sandbox_id=str(sandbox.sandbox_id or ""),
            runner_id=str(sandbox.runner_id or ""),
            profile_id=str(sandbox.profile_id or ""),
            applied_egress_mode=str(sandbox.applied_egress_mode or ""),
            applied_ingress_mode=sandbox.applied_ingress_mode,
            resource_requests=dict(sandbox.resource_requests or {}),
            resource_limits=dict(sandbox.resource_limits or {}),
        )
        if not identity.eligible:
            raise RuntimeError(
                "CoreWeave doctor attestation drift: "
                + "; ".join(identity.failures)
            )
        result = sandbox.exec(
            [
                "sh",
                "-lc",
                "test \"$(id -u)\" != 0"
                " && test ! -e /var/run/secrets/kubernetes.io/serviceaccount/token"
                " && ! touch /fugue-root-write-probe"
                " && touch /workspace/fugue-write-probe"
                " && touch /tests/fugue-write-probe"
                " && touch /solution/fugue-write-probe"
                " && touch /harbor/fugue-write-probe"
                " && rm /workspace/fugue-write-probe"
                " /tests/fugue-write-probe"
                " /solution/fugue-write-probe"
                " /harbor/fugue-write-probe",
            ],
            timeout_seconds=30,
        ).result()
        if result.returncode != 0:
            raise RuntimeError("CoreWeave non-root/read-only probe failed")
        manifest_result = sandbox.exec(
            ["cat", COREWEAVE_RUNTIME_MANIFEST_PATH],
            timeout_seconds=15,
        ).result()
        if manifest_result.returncode != 0:
            raise RuntimeError("CoreWeave runtime manifest is absent")
        try:
            manifest_raw = json.loads(str(manifest_result.stdout or ""))
            if not isinstance(manifest_raw, Mapping):
                raise ValueError("manifest must be an object")
            effective_manifest = runtime_manifest_from_dict(manifest_raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("CoreWeave runtime manifest is invalid") from exc
        if not hmac.compare_digest(
            effective_manifest.manifest_sha256,
            lock.runtime_manifest.manifest_sha256,
        ):
            raise RuntimeError("CoreWeave runtime manifest differs from its lock")
        for asset in effective_manifest.assets:
            asset_probe = sandbox.exec(
                [
                    "sh",
                    "-lc",
                    f"test -e {shlex.quote(asset.path)}"
                    f" && test ! -w {shlex.quote(asset.path)}"
                    f" && test -z \"$(find {shlex.quote(asset.path)}"
                    " -type l -print -quit)\"",
                ],
                timeout_seconds=30,
            ).result()
            if asset_probe.returncode != 0:
                raise RuntimeError(
                    f"CoreWeave runtime asset is mutable or missing: "
                    f"{asset.kind}:{asset.id}"
                )
        gateway_certificate = lock.gateway is None
        gateway_route = lock.gateway is None
        if lock.gateway is not None:
            certificate = sandbox.exec(
                ["sha256sum", COREWEAVE_GATEWAY_CA_PATH],
                timeout_seconds=15,
            ).result()
            gateway_certificate = (
                certificate.returncode == 0
                and str(certificate.stdout or "").split(maxsplit=1)[0]
                == lock.gateway.certificate_sha256
            )
            if not gateway_certificate:
                raise RuntimeError("CoreWeave gateway certificate probe failed")
            signing_key = _gateway_signing_key()
            token = mint_cell_capability(
                signing_key=signing_key,
                issuer=_identifier(
                    __import__("os").environ.get("FUGUE_INSTANCE_ID")
                    or "fugue-local",
                    "Fugue instance id",
                ),
                cell_id="coreweave-doctor",
                execution_fingerprint=lock.lock_sha256,
                route_ids=("model",),
                lifetime_seconds=120,
                max_requests=4,
                max_request_bytes=64 * 1024,
                max_response_bytes=4 * 1024 * 1024,
            )
            sandbox.write_file(
                "/tmp/fugue-doctor-capability", token.encode(), timeout_seconds=15
            ).result()
            gateway_probe = sandbox.exec(
                [
                    "python",
                    "-c",
                    _doctor_connectivity_script(
                        f"{lock.gateway.base_url.rstrip('/')}/routes/model/v1/models"
                    ),
                ],
                timeout_seconds=45,
            ).result()
            gateway_route = gateway_probe.returncode == 0
            if not gateway_route:
                raise RuntimeError("CoreWeave gateway connectivity probe failed")
        else:
            blocked_probe = sandbox.exec(
                ["python", "-c", _doctor_connectivity_script(None)],
                timeout_seconds=30,
            ).result()
            if blocked_probe.returncode != 0:
                raise RuntimeError("CoreWeave deny-all connectivity probe failed")
        return {
            "status": "passed",
            "lock_sha256": lock.lock_sha256,
            "sandbox_id": str(sandbox.sandbox_id or ""),
            "runner_id": str(sandbox.runner_id or ""),
            "profile_id": str(sandbox.profile_id or ""),
            "network": lock.network.selected_mode,
            "attestation_sha256": identity.attestation_sha256,
            "checks": {
                "exact_runner_and_profile": True,
                "no_ingress": True,
                "locked_egress": True,
                "bounded_resources": True,
                "non_root": True,
                "service_account_token_absent": True,
                "read_only_root": True,
                "writable_workspace": True,
                "writable_harbor_paths": True,
                "runtime_manifest": True,
                "gateway_certificate": gateway_certificate,
                "gateway_route": gateway_route,
                "public_destinations_blocked": True,
                "metadata_blocked": True,
                "kubernetes_service_blocked": True,
            },
        }
    finally:
        if sandbox is not None and sandbox.sandbox_id:
            __import__("cwsandbox").Sandbox.delete(
                str(sandbox.sandbox_id), missing_ok=True
            ).result()


def _gateway_signing_key() -> bytes:
    encoded = str(
        __import__("os").environ.get("FUGUE_GATEWAY_SIGNING_KEY") or ""
    )
    try:
        key = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise RuntimeError("FUGUE_GATEWAY_SIGNING_KEY must be base64") from exc
    if len(key) < 32:
        raise RuntimeError(
            "FUGUE_GATEWAY_SIGNING_KEY must decode to at least 32 bytes"
        )
    return key


def _doctor_connectivity_script(gateway_url: str | None) -> str:
    return (
        "import pathlib, ssl, urllib.request\n"
        "opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))\n"
        "def blocked(url):\n"
        "  try:\n"
        "    opener.open(url, timeout=3)\n"
        "  except Exception:\n"
        "    return True\n"
        "  return False\n"
        "blocked_urls=['https://example.com',"
        "'http://169.254.169.254/latest/meta-data',"
        "'https://kubernetes.default.svc']\n"
        "assert all(blocked(url) for url in blocked_urls)\n"
        + (
            "token_path=pathlib.Path('/tmp/fugue-doctor-capability')\n"
            "token=token_path.read_text()\n"
            "token_path.unlink()\n"
            f"request=urllib.request.Request({gateway_url!r},"
            "headers={'Authorization':'Bearer '+token})\n"
            "context=ssl.create_default_context("
            f"cafile={COREWEAVE_GATEWAY_CA_PATH!r})\n"
            "gateway_opener=urllib.request.build_opener("
            "urllib.request.ProxyHandler({}),"
            "urllib.request.HTTPSHandler(context=context))\n"
            "with gateway_opener.open(request, timeout=15) as response:\n"
            "  assert 200 <= response.status < 300\n"
            if gateway_url is not None
            else ""
        )
    )


def sweep_coreweave_orphans(
    *, instance_id: str, older_than_seconds: int
) -> dict[str, Any]:
    """Operator-only cleanup, scoped to the exact Fugue instance tag."""
    assert_coreweave_versions()
    instance = _identifier(instance_id, "Fugue instance id")
    instance_tag = f"fi-{stable_digest({'instance_id': instance})[:20]}"
    age = _positive_int(older_than_seconds, "orphan age")
    now = datetime.now(UTC)
    selected = __import__("cwsandbox").Sandbox.list(
        tags=["fugue", instance_tag],
        include_stopped=True,
    ).result()
    deleted: list[str] = []
    retained: list[str] = []
    for sandbox in selected:
        sandbox_id = str(sandbox.sandbox_id or "")
        started_at = sandbox.started_at
        if not sandbox_id or started_at is None:
            retained.append(sandbox_id or "unknown")
            continue
        if (now - started_at).total_seconds() < age:
            retained.append(sandbox_id)
            continue
        __import__("cwsandbox").Sandbox.delete(
            sandbox_id, missing_ok=True
        ).result()
        deleted.append(sandbox_id)
    return {
        "instance_id": instance,
        "instance_tag": instance_tag,
        "deleted": deleted,
        "retained": retained,
    }


try:
    from harbor.environments.capabilities import EnvironmentCapabilities
    from harbor.environments.cwsandbox import CWSandboxEnvironment
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.task.config import NetworkMode

    _HARBOR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    EnvironmentCapabilities = None  # type: ignore[assignment,misc]
    CWSandboxEnvironment = object  # type: ignore[assignment,misc]
    EnvironmentType = None  # type: ignore[assignment,misc]
    NetworkMode = None  # type: ignore[assignment,misc]
    _HARBOR_AVAILABLE = False


class FugueCoreWeaveEnvironment(CWSandboxEnvironment):  # type: ignore[misc]
    """Harbor's CoreWeave environment with Fugue profile and evidence locks."""

    _provider_label: ClassVar[str] = "fugue-coreweave"

    def __init__(
        self,
        *args: Any,
        fugue_coreweave_lock: Mapping[str, Any],
        fugue_instance_id: str,
        egress_mode: str,
        profile_ids: Sequence[str],
        profile_names: Sequence[str],
        runner_ids: Sequence[str],
        attestation_name: str = COREWEAVE_ATTESTATION_NAME,
        **kwargs: Any,
    ) -> None:
        if not _HARBOR_AVAILABLE:
            raise RuntimeError(
                "CoreWeave execution requires the fugue[coreweave] extra"
            )
        assert_coreweave_versions()
        self._fugue_lock = coreweave_lock_from_dict(fugue_coreweave_lock)
        self._fugue_instance_id = _identifier(
            fugue_instance_id, "Fugue instance id"
        )
        if tuple(profile_names) != (self._fugue_lock.profile_name,):
            raise ValueError("CoreWeave profile selector differs from its lock")
        if tuple(profile_ids) != (self._fugue_lock.profile_id,):
            raise ValueError("CoreWeave profile id differs from its lock")
        if tuple(runner_ids) != (self._fugue_lock.runner_id,):
            raise ValueError("CoreWeave runner selector differs from its lock")
        if egress_mode != self._fugue_lock.network.egress_mode:
            raise ValueError("CoreWeave egress selector differs from its lock")
        if not attestation_name or "/" in attestation_name or ".." in attestation_name:
            raise ValueError("invalid CoreWeave attestation filename")
        self._fugue_egress_mode = egress_mode
        self._fugue_profile_ids = list(profile_ids)
        self._fugue_profile_names = list(profile_names)
        self._fugue_runner_ids = list(runner_ids)
        self._fugue_attestation_name = attestation_name
        self._fugue_attestation: EffectiveSandboxAttestationV1 | None = None
        super().__init__(*args, **kwargs)
        self._fugue_operation_id = stable_digest(
            {
                "lock_sha256": self._fugue_lock.lock_sha256,
                "tags": sorted(str(value) for value in self._tags),
            }
        )
        allowed_env = {
            COREWEAVE_CAPABILITY_TOKEN_ENV,
            "FUGUE_INSTANCE_ID",
            "FUGUE_RUN_ID",
            "FUGUE_CELL_ID",
            "FUGUE_EXECUTION_FINGERPRINT",
        }
        unexpected_env = sorted(set(self._persistent_env) - allowed_env)
        if unexpected_env:
            raise ValueError(
                "CoreWeave environment contains unapproved variables: "
                + ", ".join(unexpected_env)
            )
        if self._mounts_json:
            raise ValueError("CoreWeave execution forbids host volume mounts")
        if self._secrets:
            raise ValueError("CoreWeave sandboxes receive no provider credentials")
        expected_instance_tag = (
            f"fi-{stable_digest({'instance_id': self._fugue_instance_id})[:20]}"
        )
        if expected_instance_tag not in self._tags:
            raise ValueError("CoreWeave tags do not bind the Fugue instance")

    @staticmethod
    def type() -> Any:
        return (
            EnvironmentType.CWSANDBOX
            if EnvironmentType is not None
            else "cwsandbox"
        )

    @property
    def capabilities(self) -> Any:
        return EnvironmentCapabilities(
            disable_internet=True,
            network_allowlist=True,
            network_allowlist_hostnames=True,
            network_allowlist_ipv4_addresses=True,
            network_allowlist_ipv6_addresses=True,
            network_allowlist_ipv4_cidrs=True,
            network_allowlist_ipv6_cidrs=True,
        )

    def _sandbox_kwargs(self) -> dict[str, Any]:
        kwargs = super()._sandbox_kwargs()
        if self.network_policy.network_mode == NetworkMode.PUBLIC:
            raise ValueError("unrestricted CoreWeave internet egress is forbidden")
        selected_mode = self._fugue_egress_mode
        kwargs.update(
            {
                "profile_names": self._fugue_profile_names,
                "profile_ids": self._fugue_profile_ids,
                "runner_ids": self._fugue_runner_ids,
                "network": self._sdk.NetworkOptions(
                    egress_mode=selected_mode,
                    ingress_mode=None,
                ),
                "container_image": self._fugue_lock.image,
            }
        )
        kwargs.pop("secrets", None)
        return kwargs

    async def start(self, force_build: bool) -> None:
        try:
            await self._start_attested(force_build)
        except BaseException as exc:
            detail = str(exc).lower()
            operation = (
                "attestation"
                if "attestation" in detail or "runtime manifest" in detail
                else "start"
            )
            self._write_failure(
                classify_coreweave_failure(exc, operation)
            )
            raise

    async def _start_attested(self, force_build: bool) -> None:
        self._write_operation("create_pending")
        matching = await self._sdk.Sandbox.list(
            tags=list(self._tags),
            profile_ids=self._fugue_profile_ids,
            runner_ids=self._fugue_runner_ids,
        )
        existing = [
            item
            for item in matching
            if set(str(value) for value in item.tags or ()) == set(self._tags)
        ]
        if existing:
            for sandbox in existing:
                sandbox_id = str(sandbox.sandbox_id or "")
                if sandbox_id:
                    await self._sdk.Sandbox.delete(sandbox_id, missing_ok=True)
            self._write_operation(
                "recovered_interrupted",
                sandbox_id=(
                    str(existing[0].sandbox_id or "") if len(existing) == 1 else None
                ),
            )
            raise RuntimeError(
                "recovered a tagged CoreWeave sandbox from an interrupted "
                "attempt; Fugue will not silently relaunch Agent execution"
            )
        await super().start(force_build)
        sandbox = self._require_sandbox()
        self._fugue_attestation = validate_effective_attestation(
            self._fugue_lock,
            sandbox_id=str(sandbox.sandbox_id or ""),
            runner_id=str(sandbox.runner_id or ""),
            profile_id=str(getattr(sandbox, "profile_id", None) or ""),
            applied_egress_mode=str(sandbox.applied_egress_mode or ""),
            applied_ingress_mode=getattr(sandbox, "applied_ingress_mode", None),
            resource_requests=dict(sandbox.resource_requests or {}),
            resource_limits=dict(sandbox.resource_limits or {}),
        )
        runtime_failures = await self._runtime_image_failures()
        if runtime_failures:
            unsigned = replace(
                self._fugue_attestation,
                eligible=False,
                failures=(
                    *self._fugue_attestation.failures,
                    *runtime_failures,
                ),
                attestation_sha256="",
            )
            self._fugue_attestation = replace(
                unsigned,
                attestation_sha256=stable_digest(unsigned.unsigned_dict()),
            )
        if self._fugue_lock.gateway is not None:
            certificate = await self.exec(
                f"sha256sum {COREWEAVE_GATEWAY_CA_PATH}",
                timeout_sec=15,
            )
            observed_digest = str(certificate.stdout or "").split(maxsplit=1)[0]
            if (
                certificate.return_code != 0
                or observed_digest
                != self._fugue_lock.gateway.certificate_sha256
            ):
                unsigned = replace(
                    self._fugue_attestation,
                    eligible=False,
                    failures=(
                        *self._fugue_attestation.failures,
                        "gateway certificate differs from lock",
                    ),
                    attestation_sha256="",
                )
                self._fugue_attestation = replace(
                    unsigned,
                    attestation_sha256=stable_digest(unsigned.unsigned_dict()),
                )
        if sandbox.started_at is not None:
            started = replace(
                self._fugue_attestation,
                started_at=sandbox.started_at.isoformat(),
                attestation_sha256="",
            )
            self._fugue_attestation = replace(
                started,
                attestation_sha256=stable_digest(started.unsigned_dict()),
            )
        self._write_fugue_attestation()
        if not self._fugue_attestation.eligible:
            await super().stop(delete=True)
            self._write_operation(
                "attestation_failed",
                sandbox_id=str(sandbox.sandbox_id or ""),
            )
            raise RuntimeError(
                "CoreWeave sandbox attestation drift: "
                + "; ".join(self._fugue_attestation.failures)
            )
        self._write_operation(
            "ready_for_agent",
            sandbox_id=str(sandbox.sandbox_id or ""),
        )

    async def _runtime_image_failures(self) -> tuple[str, ...]:
        manifest_result = await self.exec(
            f"cat {shlex.quote(COREWEAVE_RUNTIME_MANIFEST_PATH)}",
            timeout_sec=15,
        )
        if manifest_result.return_code != 0:
            return ("runtime manifest is absent from the locked image",)
        try:
            raw = json.loads(str(manifest_result.stdout or ""))
            if not isinstance(raw, Mapping):
                raise ValueError("manifest is not an object")
            observed = runtime_manifest_from_dict(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            return (f"runtime manifest is invalid: {str(exc)[:200]}",)
        if not hmac.compare_digest(
            observed.manifest_sha256,
            self._fugue_lock.runtime_manifest.manifest_sha256,
        ):
            return ("runtime manifest differs from the sandbox lock",)
        failures: list[str] = []
        for asset in observed.assets:
            quoted = shlex.quote(asset.path)
            probe = await self.exec(
                f"test -e {quoted} && test ! -w {quoted} "
                f"&& test -z \"$(find {quoted} -type l -print -quit)\"",
                timeout_sec=30,
            )
            if probe.return_code != 0:
                failures.append(f"runtime asset failed immutability probe: {asset.kind}:{asset.id}")
        return tuple(failures)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        target = PurePosixPath(target_dir)
        if target.parent == PurePosixPath("/harbor/skills"):
            asset = next(
                (
                    item
                    for item in self._fugue_lock.runtime_manifest.assets
                    if item.kind == "skill" and item.id == target.name
                ),
                None,
            )
            if asset is None:
                raise RuntimeError(
                    f"Skill {target.name!r} is not materialized in the locked image"
                )
            destination = shlex.quote(target.as_posix())
            source = shlex.quote(f"{asset.path}/.")
            await self._exec_checked(
                f"mkdir -p {destination} && cp -a -- {source} {destination}"
                f" && chmod -R a-w -- {destination}",
                f"stage locked Skill {target.name}",
                timeout_sec=120,
                user="root",
            )
            return
        await super().upload_dir(source_dir, target_dir)

    async def download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
    ) -> None:
        try:
            await self._download_dir_with_exclusions_bounded(
                source_dir=source_dir,
                target_dir=target_dir,
                exclude=exclude,
            )
        except BaseException as exc:
            self._write_failure(classify_coreweave_failure(exc, "artifact"))
            raise

    async def _download_dir_with_exclusions_bounded(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
    ) -> None:
        source = PurePosixPath(source_dir)
        allowed_roots = {
            PurePosixPath("/logs"),
            PurePosixPath("/workspace"),
            PurePosixPath("/tests"),
            PurePosixPath("/solution"),
            PurePosixPath("/harbor"),
        }
        if not source.is_absolute() or not any(
            source == root or source.is_relative_to(root)
            for root in allowed_roots
        ):
            raise RuntimeError(
                "CoreWeave artifact source is outside a bounded output root"
            )
        if any(
            not pattern
            or "\x00" in pattern
            or "\n" in pattern
            or pattern.startswith("/")
            for pattern in exclude
        ):
            raise RuntimeError("CoreWeave artifact exclusion is invalid")
        target = Path(target_dir)
        if target.is_symlink():
            raise RuntimeError(
                "CoreWeave artifact destination may not be a symlink"
            )
        target.mkdir(parents=True, exist_ok=True)

        remote_tar = self._new_remote_tar_path()
        async with self._remote_tar_cleanup(remote_tar):
            exclude_flags = " ".join(
                f"--exclude={shlex.quote(pattern)}" for pattern in exclude
            )
            await self._exec_checked(
                f"tar czf {shlex.quote(remote_tar)} {exclude_flags} "
                f"-C {shlex.quote(source.as_posix())} .",
                f"create bounded transfer archive for {source.as_posix()!r}",
                timeout_sec=300,
                user="root",
            )
            size = await self._exec_checked(
                f"stat -c %s -- {shlex.quote(remote_tar)}",
                "measure CoreWeave transfer archive",
                timeout_sec=15,
                user="root",
            )
            try:
                archive_bytes = int(str(size.stdout or "").strip())
            except ValueError as exc:
                raise RuntimeError(
                    "CoreWeave transfer archive has an invalid size"
                ) from exc
            if (
                archive_bytes < 0
                or archive_bytes > _MAX_TRANSFER_ARCHIVE_BYTES
            ):
                raise RuntimeError(
                    "CoreWeave transfer archive exceeds the compressed byte limit"
                )

            with tempfile.TemporaryDirectory() as temporary:
                archive_path = Path(temporary) / "transfer.tar.gz"
                await super().download_file(remote_tar, archive_path)
                members = validate_coreweave_artifact_archive(archive_path)
                try:
                    with tarfile.open(archive_path, "r:gz") as archive:
                        archive.extractall(
                            path=target,
                            members=members,
                            filter="data",
                        )
                except (OSError, tarfile.TarError) as exc:
                    raise RuntimeError(
                        "CoreWeave artifact archive could not be extracted"
                    ) from exc

    async def _start_sdk_sandbox(self, sandbox: Any) -> None:
        await super()._start_sdk_sandbox(sandbox)
        self._write_operation(
            "create_accepted",
            sandbox_id=str(sandbox.sandbox_id or ""),
        )

    async def stop(self, delete: bool) -> None:
        stopped_at = datetime.now(UTC).isoformat()
        sandbox = self._sandbox
        sandbox_id = str(sandbox.sandbox_id or "") if sandbox is not None else None
        cleanup_failure: BaseException | None = None
        try:
            await super().stop(delete=delete)
        except BaseException as exc:
            cleanup_failure = exc
            self._write_failure(
                classify_coreweave_failure(exc, "delete" if delete else "stop")
            )
        if self._fugue_attestation is not None:
            failures = list(self._fugue_attestation.failures)
            if cleanup_failure is not None:
                failures.append("sandbox teardown failed")
            elif not delete:
                failures.append("sandbox was stopped but not deleted")
            unsigned = replace(
                self._fugue_attestation,
                stopped_at=stopped_at,
                deleted=delete and cleanup_failure is None,
                eligible=not failures,
                failures=tuple(dict.fromkeys(failures)),
                attestation_sha256="",
            )
            self._fugue_attestation = replace(
                unsigned,
                attestation_sha256=stable_digest(unsigned.unsigned_dict()),
            )
            self._write_fugue_attestation()
        self._write_operation(
            "cleanup_failed" if cleanup_failure is not None else "deleted",
            sandbox_id=sandbox_id,
        )
        if cleanup_failure is not None:
            raise cleanup_failure

    def _write_fugue_attestation(self) -> None:
        if self._fugue_attestation is None:
            return
        destination = (
            Path(self.trial_paths.artifacts_dir) / self._fugue_attestation_name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination, self._fugue_attestation.to_dict())

    def _write_operation(
        self, state: str, *, sandbox_id: str | None = None
    ) -> None:
        unsigned = {
            "schema_version": 1,
            "operation_id": self._fugue_operation_id,
            "state": state,
            "sandbox_id": sandbox_id,
            "lock_sha256": self._fugue_lock.lock_sha256,
            "profile_id": self._fugue_lock.profile_id,
            "runner_id": self._fugue_lock.runner_id,
            "tags": sorted(str(value) for value in self._tags),
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        destination = (
            Path(self.trial_paths.artifacts_dir) / COREWEAVE_OPERATION_NAME
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            destination,
            {**unsigned, "record_sha256": stable_digest(unsigned)},
        )

    def _write_failure(self, failure: CoreWeaveFailureV1) -> None:
        destination = (
            Path(self.trial_paths.artifacts_dir) / COREWEAVE_FAILURE_NAME
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination, failure.to_dict())


def _reject_unknown(
    raw: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return {str(k): v for k, v in value.items()}


def _string_mapping(value: Any, label: str) -> dict[str, str]:
    return {
        str(key): _nonempty(item, f"{label} value")
        for key, item in _mapping(value, label).items()
    }


def _nonempty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _identifier(value: Any, label: str) -> str:
    selected = _nonempty(value, label)
    if len(selected) > 160 or not all(
        character.isalnum() or character in "._:-" for character in selected
    ):
        raise ValueError(f"invalid {label}")
    return selected


def _digest_image(value: Any) -> str:
    image = _nonempty(value, "CoreWeave image")
    if not _DIGEST_IMAGE.fullmatch(image):
        raise ValueError("CoreWeave image must be pinned by sha256 digest")
    return image


def _safe_runtime_path(value: Any) -> str:
    selected = PurePosixPath(_nonempty(value, "runtime asset path"))
    if not selected.is_absolute() or ".." in selected.parts:
        raise ValueError("runtime asset path must be absolute and normalized")
    allowed_roots = (
        PurePosixPath("/opt/fugue"),
        PurePosixPath("/fugue-components"),
        PurePosixPath("/fugue-src"),
    )
    if not any(selected == root or root in selected.parents for root in allowed_roots):
        raise ValueError("runtime asset path must stay in a locked image asset root")
    return selected.as_posix()


def _text_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("expected a sequence of strings")
    normalized = tuple(_nonempty(item, "sequence value") for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("sequence values must be unique")
    return normalized


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        selected = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if selected < 1:
        raise ValueError(f"{label} must be a positive integer")
    return selected


def _sha256(value: Any, label: str) -> str:
    selected = _nonempty(value, label)
    if not re.fullmatch(r"[0-9a-f]{64}", selected):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return selected


def _https_url(value: Any) -> str:
    from urllib.parse import urlparse

    selected = _nonempty(value, "gateway base URL").rstrip("/")
    parsed = urlparse(selected)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("gateway base URL must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("gateway base URL may not contain credentials or query data")
    return selected
