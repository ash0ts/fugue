from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from typing import Any

from fugue.bench.distribution_assets import runtime_assets, vendor_asset
from fugue.bench.executables import resolve_console_script
from fugue.bench.runtime_provenance import (
    resolve_fugue_distribution_provenance,
    resolve_workspace_source_provenance,
)
from fugue.model_plane import (
    missing_model_env,
    provider_api_key_env,
    resolve_model_route,
)
from fugue.resource_integrity import verify_packaged_assets

DOCTOR_SCHEMA_VERSION = 1
_RUNTIME_ASSET_GROUPS = (
    "claude-code",
    "codex",
    "fugue-context",
    "gitnexus",
    "hermes",
    "openclaw",
)
_SUPPORTED_REQUIRED_CAPABILITIES = frozenset({"local-runner"})
_MINIMUM_LOCAL_RUNNER_DISK_BYTES = 10 * 1024**3


def doctor_report(
    workspace: Path | None = None,
    *,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
    required_capabilities: Iterable[str] = (),
) -> dict[str, Any]:
    """Inspect Fugue, optionally failing closed for requested capabilities.

    With no requested capability the report preserves the observational core-install
    contract: optional execution dependencies and credentials are reported but do not
    affect ``ok``. A requested capability turns ``ok`` into a strict readiness gate.
    """

    root = (workspace or Path.cwd()).resolve()
    values = env if env is not None else os.environ
    requested = _normalize_required_capabilities(required_capabilities)
    runtime_groups = {
        component_id: {
            "files": len(assets),
            "available": bool(assets),
        }
        for component_id in _RUNTIME_ASSET_GROUPS
        if (assets := runtime_assets(component_id)) is not None
    }
    vendor_available = _vendor_available()
    context_root = files("fugue").joinpath("resources", "context-systems")
    context_ids = sorted(
        item.name.removesuffix(".yaml")
        for item in context_root.iterdir()
        if item.is_file() and item.name.endswith(".yaml")
    )
    schema_root = files("fugue").joinpath("resources", "schemas")
    schema_count = _resource_file_count(schema_root)
    asset_integrity = verify_packaged_assets()
    missing_runtime_groups = sorted(
        set(_RUNTIME_ASSET_GROUPS) - set(runtime_groups)
    )
    unavailable_runtime_groups = sorted(
        component_id
        for component_id, item in runtime_groups.items()
        if not item["available"]
    )
    required_asset_failures = [
        *(f"missing runtime group {item}" for item in missing_runtime_groups),
        *(f"empty runtime group {item}" for item in unavailable_runtime_groups),
        *(() if vendor_available else ("missing vendor archive",)),
        *(() if "none" in context_ids else ("missing none context system",)),
        *(() if schema_count > 0 else ("missing provider schemas",)),
    ]
    assets_ok = asset_integrity["ready"] and not required_asset_failures
    optional = {
        "weave": _optional_package("weave"),
        "local_runner": _optional_package("harbor"),
    }
    docker = _docker_status(probe_network="local-runner" in requested)
    optional["local_runner"]["required_version"] = "0.18.0"
    optional["local_runner"]["version_compatible"] = (
        optional["local_runner"]["version"] == "0.18.0"
    )
    optional["local_runner"]["docker_cli"] = docker["cli_available"]
    optional["local_runner"]["docker_daemon"] = docker["daemon_available"]
    optional["local_runner"]["executable"] = resolve_console_script("harbor")
    optional["local_runner"]["ready"] = bool(
        optional["local_runner"]["installed"]
        and optional["local_runner"]["version_compatible"]
        and optional["local_runner"]["executable"]
        and docker["daemon_available"]
        and docker.get("network_ready", True)
    )
    route = resolve_model_route(model, values) if model is not None else None
    credential_names = missing_model_env(route, values) if route is not None else ()
    credentials = {
        name: bool(values.get(name))
        for name in ("ANTHROPIC_API_KEY", "WANDB_API_KEY")
    }
    architecture = _architecture()
    free_disk = shutil.disk_usage(root).free
    python_version_info = _python_version_info()
    python_version = ".".join(str(part) for part in python_version_info)
    python_supported = (3, 12) <= python_version_info < (3, 14)
    local_runner_supported = (3, 13) <= python_version_info < (3, 14)
    readiness_requirements = {
        "packaged_assets": {
            "ready": assets_ok,
            "detail": _asset_integrity_detail(
                asset_integrity,
                required_asset_failures=required_asset_failures,
            ),
        }
    }
    if "local-runner" in requested:
        local_runner = optional["local_runner"]
        readiness_requirements.update(
            {
                "python_local_runner": {
                    "ready": local_runner_supported,
                    "detail": (
                        f"Python {python_version}; local execution requires "
                        ">=3.13,<3.14"
                    ),
                },
                "host_architecture": {
                    "ready": architecture in {"amd64", "arm64"},
                    "detail": f"observed {architecture}; supported: amd64, arm64",
                },
                "harbor_installed": {
                    "ready": bool(local_runner["installed"]),
                    "detail": (
                        f"Harbor {local_runner['version']}"
                        if local_runner["version"]
                        else "install fugue[local-runner]"
                    ),
                },
                "harbor_version": {
                    "ready": bool(local_runner["version_compatible"]),
                    "detail": (
                        f"observed {local_runner['version'] or 'not installed'}; "
                        f"required {local_runner['required_version']}"
                    ),
                },
                "docker_cli": {
                    "ready": bool(docker["cli_available"]),
                    "detail": docker["detail"],
                },
                "docker_daemon": {
                    "ready": bool(docker["daemon_available"]),
                    "detail": docker["detail"],
                },
                "docker_network": {
                    "ready": bool(docker.get("network_ready")),
                    "detail": str(
                        docker.get("network_detail")
                        or "Docker network probe was not completed"
                    ),
                },
                "disk_space": {
                    "ready": free_disk >= _MINIMUM_LOCAL_RUNNER_DISK_BYTES,
                    "detail": (
                        f"{round(free_disk / (1024**3), 2)} GiB free; "
                        "local execution requires at least 10 GiB"
                    ),
                },
            }
        )
        if route is not None:
            readiness_requirements["model_credential"] = {
                "ready": not credential_names,
                "detail": (
                    f"{route.display_model}: credential present"
                    if not credential_names
                    else (
                        f"{route.display_model}: missing "
                        + ", ".join(credential_names)
                    )
                ),
            }
    readiness_ok = all(
        bool(requirement["ready"])
        for requirement in readiness_requirements.values()
    )
    return {
        "schema_version": DOCTOR_SCHEMA_VERSION,
        "ok": readiness_ok,
        "readiness": {
            "mode": "required" if requested else "observational",
            "requested_capabilities": list(requested),
            "ready": readiness_ok,
            "requirements": readiness_requirements,
        },
        "python": {
            "version": python_version,
            "supported": python_supported,
            "local_runner_supported": local_runner_supported,
        },
        "host": {
            "architecture": architecture,
            "architecture_supported": architecture in {"amd64", "arm64"},
            "free_disk_bytes": free_disk,
            "free_disk_gib": round(free_disk / (1024**3), 2),
            "docker": docker,
        },
        "model_route": {
            "selected": route is not None,
            "model": route.display_model if route is not None else None,
            "provider": route.provider if route is not None else None,
            "credential_name": (
                provider_api_key_env(route) if route is not None else None
            ),
            "credential_present": (
                not credential_names if route is not None else None
            ),
            "missing_credentials": list(credential_names),
        },
        "distribution": resolve_fugue_distribution_provenance(),
        "workspace": resolve_workspace_source_provenance(root),
        "assets": {
            "integrity": asset_integrity,
            "required_asset_failures": required_asset_failures,
            "runtime_groups": runtime_groups,
            "vendor_archive": vendor_available,
            "context_systems": context_ids,
            "schemas": schema_count,
        },
        "optional_features": optional,
        "credentials_present": credentials,
    }


def _asset_integrity_detail(
    integrity: Mapping[str, Any],
    *,
    required_asset_failures: Iterable[str] = (),
) -> str:
    verified = int(integrity["verified_files"])
    expected = int(integrity["expected_files"])
    required_failures = tuple(required_asset_failures)
    if integrity["ready"] and not required_failures:
        return f"verified {verified}/{expected} exact packaged static files"
    failures = []
    for key, label in (
        ("manifest_errors", "manifest errors"),
        ("missing_files", "missing"),
        ("tampered_files", "changed"),
        ("unexpected_files", "unexpected"),
        ("unsafe_files", "unsafe"),
    ):
        if count := len(integrity[key]):
            failures.append(f"{count} {label}")
    failures.extend(required_failures)
    return f"verified {verified}/{expected}; " + ", ".join(failures)


def _normalize_required_capabilities(values: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(sorted({str(value).strip() for value in values if str(value).strip()}))
    unsupported = sorted(set(requested) - _SUPPORTED_REQUIRED_CAPABILITIES)
    if unsupported:
        raise ValueError(
            "unsupported required doctor capabilities: " + ", ".join(unsupported)
        )
    return requested


def _python_version_info() -> tuple[int, int, int]:
    return tuple(sys.version_info[:3])


def _vendor_available() -> bool:
    try:
        return bool(vendor_asset("weave-node-sdk.tgz").body)
    except FileNotFoundError:
        return False


def _optional_package(distribution: str) -> dict[str, Any]:
    try:
        installed_version = version(distribution)
    except PackageNotFoundError:
        installed_version = None
    try:
        importable = importlib.util.find_spec(distribution) is not None
    except (ImportError, ValueError):
        importable = False
    return {
        "installed": installed_version is not None and importable,
        "version": installed_version,
    }


def _architecture() -> str:
    observed = platform.machine().lower()
    return {"x86_64": "amd64", "aarch64": "arm64"}.get(observed, observed)


def _docker_status(*, probe_network: bool = False) -> dict[str, Any]:
    executable = shutil.which("docker")
    if executable is None:
        return {
            "cli_available": False,
            "daemon_available": False,
            "detail": "docker CLI not found",
            "network_ready": False,
            "network_detail": "docker CLI not found",
        }
    try:
        result = subprocess.run(
            [executable, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "cli_available": True,
            "daemon_available": False,
            "detail": str(exc),
            "network_ready": False,
            "network_detail": "Docker daemon was unavailable",
        }
    detail = (result.stdout if result.returncode == 0 else result.stderr).strip()
    daemon_available = result.returncode == 0
    network_ready = daemon_available
    network_detail = (
        "not probed; strict local-runner readiness was not requested"
    )
    if daemon_available and probe_network:
        network_ready, network_detail = _probe_docker_network(executable)
    elif not daemon_available:
        network_detail = "Docker daemon was unavailable"
    return {
        "cli_available": True,
        "daemon_available": daemon_available,
        "detail": detail or f"docker info exited {result.returncode}",
        "network_ready": network_ready,
        "network_detail": network_detail,
    }


def _probe_docker_network(executable: str) -> tuple[bool, str]:
    """Verify the Docker network operations that Harbor requires."""

    name = f"fugue-doctor-probe-{uuid.uuid4().hex[:12]}"
    try:
        created = subprocess.run(
            [
                executable,
                "network",
                "create",
                "--label",
                "com.wandb.fugue.purpose=doctor-probe",
                name,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Docker network probe failed: {exc}"
    if created.returncode != 0:
        detail = (created.stderr or created.stdout).strip()
        return False, detail or "Docker could not create a probe network"
    try:
        removed = subprocess.run(
            [executable, "network", "rm", name],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"probe network {name} was created but cleanup failed: {exc}"
    if removed.returncode != 0:
        detail = (removed.stderr or removed.stdout).strip()
        return (
            False,
            detail or f"probe network {name} was created but not removed",
        )
    return True, "created and removed an isolated Docker probe network"


def _resource_file_count(root: Any) -> int:
    if not root.is_dir():
        return 0
    count = 0
    for item in root.iterdir():
        if item.is_dir():
            count += _resource_file_count(item)
        elif item.is_file():
            count += 1
    return count
