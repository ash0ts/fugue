from __future__ import annotations

import base64
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

from fugue.bridge import bridge_status
from fugue.model_plane import (
    ModelRoute,
    missing_model_env,
    missing_trace_env,
    provider_request_headers,
    resolve_harness_model_route,
    resolve_model_route,
    trace_project_slug,
)
from fugue.weave_support import resolved_weave_trace_server_url

HARBOR_VERSION = "0.18.0"

_HARBOR_CONFIG_VALIDATOR = """
import json
import sys
from importlib.metadata import version
from harbor.models.job.config import JobConfig

expected = sys.argv[1]
actual = version("harbor")
if actual != expected:
    raise RuntimeError(f"harbor=={expected} required; found {actual}")
for value in sys.argv[2:]:
    with open(value) as handle:
        config = json.load(handle)
    config.pop("fugue", None)
    unknown = sorted(set(config) - set(JobConfig.model_fields))
    if unknown:
        raise ValueError("unknown Harbor fields: " + ", ".join(unknown))
    JobConfig.model_validate(config)
"""


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


def run_preflight(
    model: str | None = None,
    *,
    repo_root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    live: bool = True,
    harnesses: tuple[str, ...] | None = None,
    builder_model: str | None = None,
    judge_model: str | None = None,
) -> list[PreflightCheck]:
    values = env if env is not None else os.environ
    root = Path.cwd() if repo_root is None else Path(repo_root)
    checks: list[PreflightCheck] = []

    try:
        route = resolve_model_route(model, env=values)
    except ValueError as exc:
        return [PreflightCheck("model route", False, str(exc))]

    checks.append(
        PreflightCheck(
            "model route",
            True,
            f"{route.display_model} via {route.provider} at {_route_endpoint(route)}",
        )
    )
    _append_env_checks(checks, route, values)
    _append_local_tool_checks(checks, root)

    if not live:
        return checks

    if not missing_model_env(route, values):
        checks.append(_check_provider_metadata(route, values))
    if not missing_trace_env(values):
        checks.append(_check_weave_endpoint(values))

    bridge_required = harnesses is None or any(
        resolve_harness_model_route(route, harness)["bridge_required"]
        for harness in harnesses
    )
    if not bridge_required:
        checks.append(
            PreflightCheck(
                "bridge health",
                True,
                "not required; every selected harness uses its provider endpoint directly",
            )
        )
        return checks
    status = (
        bridge_status()
        if harnesses is None
        else bridge_status(
            repo_root=root,
            route=route,
            builder_route=(
                resolve_model_route(builder_model, values) if builder_model else None
            ),
            judge_route=(
                resolve_model_route(judge_model, values) if judge_model else None
            ),
            env=values,
        )
    )
    checks.append(
        PreflightCheck(
            "bridge health",
            bool(status.get("ok")),
            str(
                status.get("error")
                or (
                    f"locked {status['runtime_lock']['image']} as "
                    f"{status['resolved_image_id']}"
                    if status.get("runtime_lock") and status.get("resolved_image_id")
                    else status.get("body") or status
                )
            ),
        )
    )
    return checks


def _route_endpoint(route: ModelRoute) -> str:
    return str(
        route.responses_base_url
        or route.messages_base_url
        or route.chat_base_url
        or "unavailable"
    )


def _append_env_checks(
    checks: list[PreflightCheck], route: ModelRoute, env: Mapping[str, str]
) -> None:
    trace_missing = missing_trace_env(env)
    checks.append(
        PreflightCheck(
            "trace env",
            not trace_missing,
            f"WANDB trace env present; target {trace_project_slug(env)}"
            if not trace_missing
            else "missing " + ", ".join(trace_missing),
        )
    )
    model_missing = missing_model_env(route, env)
    checks.append(
        PreflightCheck(
            "model env",
            not model_missing,
            f"{route.api_key_env} present"
            if not model_missing
            else "missing " + ", ".join(model_missing),
        )
    )


def _append_local_tool_checks(
    checks: list[PreflightCheck], repo_root: Path | str
) -> None:
    checks.append(
        PreflightCheck(
            "docker",
            shutil.which("docker") is not None,
            "docker CLI found" if shutil.which("docker") else "docker CLI not found",
        )
    )
    harbor = shutil.which("harbor")
    checks.append(
        PreflightCheck(
            "harbor",
            harbor is not None,
            f"harbor found at {harbor}" if harbor else "harbor CLI not found",
        )
    )
    checks.append(harbor_version_check())
    checks.append(harbor_import_check(repo_root))


def harbor_version_check() -> PreflightCheck:
    harbor = shutil.which("harbor")
    if not harbor:
        return PreflightCheck(
            "harbor version", False, f"harbor=={HARBOR_VERSION} is required"
        )
    harbor_py = Path(harbor).resolve().parent / "python"
    if not harbor_py.exists():
        return PreflightCheck("harbor version", False, f"{harbor_py} not found")
    command = [
        harbor_py.as_posix(),
        "-c",
        "from importlib.metadata import version; print(version('harbor'))",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    actual = result.stdout.strip() if result.returncode == 0 else ""
    ok = actual == HARBOR_VERSION
    return PreflightCheck(
        "harbor version",
        ok,
        f"harbor=={actual} found"
        if ok
        else f"harbor=={HARBOR_VERSION} required; found {actual or 'unknown'}",
    )


def validate_harbor_job_configs(paths: list[Path]) -> None:
    """Validate rendered JSON with the exact Python environment Harbor will use."""
    if not paths:
        return
    harbor = shutil.which("harbor")
    if not harbor:
        raise RuntimeError(f"harbor=={HARBOR_VERSION} CLI is required")
    harbor_py = Path(harbor).resolve().parent / "python"
    if not harbor_py.exists():
        raise RuntimeError(f"Harbor Python not found: {harbor_py}")
    command = [
        harbor_py.as_posix(),
        "-c",
        _HARBOR_CONFIG_VALIDATOR,
        HARBOR_VERSION,
        *[path.as_posix() for path in paths],
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Harbor JobConfig validation failed: {detail}")


def _check_provider_metadata(
    route: ModelRoute, env: Mapping[str, str], timeout_sec: float = 20.0
) -> PreflightCheck:
    headers = {
        "Authorization": f"Bearer {env.get(route.api_key_env, '')}",
        **provider_request_headers(route, env),
    }
    if route.provider == "anthropic":
        return PreflightCheck(
            "provider metadata",
            True,
            "Anthropic key present; live model call skipped",
        )
    if not route.chat_base_url:
        return PreflightCheck("provider metadata", True, "no direct metadata endpoint")
    try:
        response = httpx.get(
            f"{route.chat_base_url}/models", headers=headers, timeout=timeout_sec
        )
    except httpx.HTTPError as exc:
        return PreflightCheck("provider metadata", False, str(exc))
    ok = response.status_code < 400
    return PreflightCheck(
        "provider metadata",
        ok,
        f"GET /models returned {response.status_code}",
    )


def _check_weave_endpoint(
    env: Mapping[str, str], timeout_sec: float = 20.0
) -> PreflightCheck:
    endpoint = resolved_weave_trace_server_url(env)
    token = base64.b64encode(f"api:{env.get('WANDB_API_KEY', '')}".encode()).decode()
    try:
        response = httpx.get(
            f"{endpoint}/server_info",
            headers={"Authorization": f"Basic {token}"},
            timeout=timeout_sec,
        )
    except httpx.HTTPError as exc:
        return PreflightCheck("weave endpoint", False, f"{endpoint}: {exc}")
    return PreflightCheck(
        "weave endpoint",
        response.status_code < 400,
        f"{endpoint}/server_info returned {response.status_code}",
    )


def harbor_import_check(repo_root: Path | str) -> PreflightCheck:
    harbor = shutil.which("harbor")
    if not harbor:
        return PreflightCheck("adapters", False, "harbor CLI not found")
    harbor_py = Path(harbor).resolve().parent / "python"
    if not harbor_py.exists():
        return PreflightCheck("adapters", False, f"{harbor_py} not found")
    result = subprocess.run(
        [harbor_py.as_posix(), "-c", "import fugue.agents"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return PreflightCheck(
        "adapters",
        result.returncode == 0,
        "fugue.agents importable under harbor python"
        if result.returncode == 0
        else (result.stderr.strip() or result.stdout.strip()),
    )
