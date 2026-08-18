from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from typing import Any

from fugue.bench.distribution_assets import runtime_assets, vendor_asset
from fugue.bench.runtime_provenance import (
    resolve_fugue_distribution_provenance,
    resolve_workspace_source_provenance,
)
from fugue.model_plane import (
    missing_model_env,
    provider_api_key_env,
    resolve_model_route,
)

DOCTOR_SCHEMA_VERSION = 1
_RUNTIME_ASSET_GROUPS = ("claude-code", "codex", "gitnexus", "hermes", "openclaw")


def doctor_report(
    workspace: Path | None = None,
    *,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect a Fugue installation without mutating or requiring optional extras."""

    root = (workspace or Path.cwd()).resolve()
    values = env if env is not None else os.environ
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
    assets_ok = (
        all(item["available"] for item in runtime_groups.values())
        and len(runtime_groups) == len(_RUNTIME_ASSET_GROUPS)
        and vendor_available
        and "none" in context_ids
        and schema_count > 0
    )
    optional = {
        "weave": _optional_package("weave"),
        "local_runner": _optional_package("harbor"),
    }
    docker = _docker_status()
    optional["local_runner"]["required_version"] = "0.18.0"
    optional["local_runner"]["version_compatible"] = (
        optional["local_runner"]["version"] == "0.18.0"
    )
    optional["local_runner"]["docker_cli"] = docker["cli_available"]
    optional["local_runner"]["docker_daemon"] = docker["daemon_available"]
    optional["local_runner"]["ready"] = bool(
        optional["local_runner"]["installed"]
        and optional["local_runner"]["version_compatible"]
        and docker["daemon_available"]
    )
    route = resolve_model_route(model, values)
    credential_names = missing_model_env(route, values)
    credentials = {
        name: bool(values.get(name))
        for name in ("ANTHROPIC_API_KEY", "WANDB_API_KEY")
    }
    architecture = _architecture()
    free_disk = shutil.disk_usage(root).free
    return {
        "schema_version": DOCTOR_SCHEMA_VERSION,
        "ok": assets_ok,
        "python": {
            "version": ".".join(str(part) for part in sys.version_info[:3]),
            "supported": sys.version_info >= (3, 12),
        },
        "host": {
            "architecture": architecture,
            "architecture_supported": architecture in {"amd64", "arm64"},
            "free_disk_bytes": free_disk,
            "free_disk_gib": round(free_disk / (1024**3), 2),
            "docker": docker,
        },
        "model_route": {
            "model": route.display_model,
            "provider": route.provider,
            "credential_name": provider_api_key_env(route),
            "credential_present": not credential_names,
            "missing_credentials": list(credential_names),
        },
        "distribution": resolve_fugue_distribution_provenance(),
        "workspace": resolve_workspace_source_provenance(root),
        "assets": {
            "runtime_groups": runtime_groups,
            "vendor_archive": vendor_available,
            "context_systems": context_ids,
            "schemas": schema_count,
        },
        "optional_features": optional,
        "credentials_present": credentials,
    }


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


def _docker_status() -> dict[str, Any]:
    executable = shutil.which("docker")
    if executable is None:
        return {
            "cli_available": False,
            "daemon_available": False,
            "detail": "docker CLI not found",
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
        }
    detail = (result.stdout if result.returncode == 0 else result.stderr).strip()
    return {
        "cli_available": True,
        "daemon_available": result.returncode == 0,
        "detail": detail or f"docker info exited {result.returncode}",
    }


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
