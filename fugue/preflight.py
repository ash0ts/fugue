from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from fugue.bridge import bridge_status, bridge_up
from fugue.model_plane import (
    ModelRoute,
    missing_model_env,
    missing_trace_env,
    resolve_model_route,
    trace_project_slug,
)


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_preflight(
    model: str | None = None,
    *,
    repo_root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    live: bool = True,
    start_bridge: bool = True,
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
            f"{route.display_model} via {route.provider}",
        )
    )
    _append_env_checks(checks, route, values)
    _append_local_tool_checks(checks, root)

    if not live:
        return checks

    if not missing_model_env(route, values):
        checks.append(_check_provider_metadata(route, values))

    if start_bridge:
        try:
            bridge_up(
                route.display_model,
                repo_root=root,
                env=values,
                builder_model=builder_model,
                judge_model=judge_model,
            )
            checks.append(PreflightCheck("bridge up", True, "docker compose is up"))
        except Exception as exc:
            checks.append(PreflightCheck("bridge up", False, str(exc)))
    status = bridge_status()
    checks.append(
        PreflightCheck(
            "bridge health",
            bool(status.get("ok")),
            str(status.get("body") or status.get("error") or status),
        )
    )
    return checks


def print_preflight(checks: list[PreflightCheck]) -> int:
    passed = 0
    failed = 0
    print("== fugue preflight ==")
    for check in checks:
        if check.ok:
            passed += 1
            marker = "[ok]  "
        else:
            failed += 1
            marker = "[FAIL]"
        print(f"  {marker} {check.name}: {check.detail}")
    print(f"== {passed} ok, {failed} failed ==")
    return 0 if failed == 0 else 1


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
    checks.append(harbor_import_check(repo_root))


def _check_provider_metadata(
    route: ModelRoute, env: Mapping[str, str], timeout_sec: float = 20.0
) -> PreflightCheck:
    headers = {"Authorization": f"Bearer {env.get(route.api_key_env, '')}"}
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
