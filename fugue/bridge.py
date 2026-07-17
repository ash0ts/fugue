from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from fugue.bench.files import atomic_write_json
from fugue.model_plane import (
    BRIDGE_MASTER_KEY_ENV,
    ModelRoute,
    model_route_identity,
    provider_request_headers,
    resolve_model_route,
)

BRIDGE_PORT = 4000
BRIDGE_RUNTIME_DIR = Path(".fugue") / "bridge"
BRIDGE_CONFIG_NAME = "litellm.config.yaml"
BRIDGE_COMPOSE_NAME = "docker-compose.yaml"
BRIDGE_LOCK_NAME = "runtime-lock.json"
LITELLM_IMAGE = (
    "ghcr.io/berriai/litellm@"
    "sha256:66a108711edea25ef531a74764f001d0c1934cc8abb422f5a3bd17a2860e4035"
)


@dataclass(frozen=True)
class BridgeFiles:
    runtime_dir: Path
    config_path: Path
    compose_path: Path
    lock_path: Path


def _litellm_params(
    route: ModelRoute,
    *,
    concrete: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": (
            route.litellm_model.replace("*", route.model_id)
            if concrete
            else route.litellm_model
        ),
        "api_key": f"os.environ/{route.api_key_env}",
    }
    if route.provider in {"wandb", "openai"} and route.chat_base_url:
        params["api_base"] = route.chat_base_url
    if route.provider == "anthropic" and route.messages_base_url:
        params["api_base"] = route.messages_base_url
    headers = provider_request_headers(route, env)
    if headers:
        params["extra_headers"] = headers
    return params


def litellm_config_for_route(
    route: ModelRoute,
    *,
    builder_route: ModelRoute | None = None,
    judge_route: ModelRoute | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    roles = {
        "fugue-target": route,
        "fugue-builder": builder_route or route,
        "fugue-judge": judge_route or route,
    }
    return {
        "model_list": [
            {
                "model_name": "*",
                "litellm_params": _litellm_params(route, env=env),
            },
            *[
                {
                    "model_name": name,
                    "litellm_params": _litellm_params(
                        role_route, concrete=True, env=env
                    ),
                }
                for name, role_route in roles.items()
            ],
        ],
        "litellm_settings": {
            "drop_params": True,
        },
        "general_settings": {
            "master_key": f"os.environ/{BRIDGE_MASTER_KEY_ENV}",
        },
    }


def docker_compose_for_route(
    route: ModelRoute,
    *,
    builder_route: ModelRoute | None = None,
    judge_route: ModelRoute | None = None,
) -> dict[str, Any]:
    routes = (route, builder_route or route, judge_route or route)
    provider_env = {item.api_key_env: f"${{{item.api_key_env}}}" for item in routes}
    return {
        "services": {
            "bridge": {
                "image": LITELLM_IMAGE,
                "container_name": "fugue-litellm-bridge",
                "ports": [f"127.0.0.1:{BRIDGE_PORT}:4000"],
                "volumes": [f"./{BRIDGE_CONFIG_NAME}:/app/config.yaml:ro"],
                "environment": {
                    **provider_env,
                    BRIDGE_MASTER_KEY_ENV: f"${{{BRIDGE_MASTER_KEY_ENV}:-sk-fugue-local}}",
                },
                "command": [
                    "--config",
                    "/app/config.yaml",
                    "--port",
                    "4000",
                    "--num_workers",
                    "4",
                ],
                "restart": "unless-stopped",
            }
        }
    }


def bridge_runtime_lock_for_route(
    route: ModelRoute,
    *,
    builder_route: ModelRoute | None = None,
    judge_route: ModelRoute | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    config = litellm_config_for_route(
        route,
        builder_route=builder_route,
        judge_route=judge_route,
        env=env,
    )
    return {
        "schema_version": 1,
        "image": LITELLM_IMAGE,
        "config_sha256": _json_digest(config),
        "target_route": model_route_identity(route),
    }


def write_bridge_files(
    route: ModelRoute,
    repo_root: Path | str | None = None,
    *,
    builder_route: ModelRoute | None = None,
    judge_route: ModelRoute | None = None,
    env: Mapping[str, str] | None = None,
) -> BridgeFiles:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    runtime_dir = root / BRIDGE_RUNTIME_DIR
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_path = runtime_dir / BRIDGE_CONFIG_NAME
    compose_path = runtime_dir / BRIDGE_COMPOSE_NAME
    lock_path = runtime_dir / BRIDGE_LOCK_NAME
    config = litellm_config_for_route(
        route, builder_route=builder_route, judge_route=judge_route, env=env
    )
    compose = docker_compose_for_route(
        route, builder_route=builder_route, judge_route=judge_route
    )
    for path, value in ((config_path, config), (compose_path, compose)):
        path.write_text(yaml.safe_dump(value, sort_keys=False))
    atomic_write_json(
        lock_path,
        bridge_runtime_lock_for_route(
            route,
            builder_route=builder_route,
            judge_route=judge_route,
            env=env,
        ),
    )
    return BridgeFiles(runtime_dir, config_path, compose_path, lock_path)


def bridge_up(
    model: str | None = None,
    *,
    repo_root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    builder_model: str | None = None,
    judge_model: str | None = None,
) -> BridgeFiles:
    route = resolve_model_route(model, env=env)
    builder_route = (
        resolve_model_route(builder_model, env=env) if builder_model else None
    )
    judge_route = resolve_model_route(judge_model, env=env) if judge_model else None
    files = write_bridge_files(
        route,
        repo_root,
        builder_route=builder_route,
        judge_route=judge_route,
        env=env,
    )
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            files.compose_path.as_posix(),
            "up",
            "-d",
            "--force-recreate",
        ],
        cwd=Path.cwd() if repo_root is None else Path(repo_root),
        check=True,
        env=dict(env) if env is not None else None,
    )
    deadline = time.monotonic() + 60
    status: dict[str, Any] = {"ok": False, "error": "bridge did not start"}
    while time.monotonic() < deadline:
        status = bridge_status(
            timeout_sec=2,
            repo_root=repo_root,
            route=route,
            builder_route=builder_route,
            judge_route=judge_route,
            env=env,
        )
        if status.get("ok") is True:
            return files
        time.sleep(0.25)
    raise RuntimeError(f"LiteLLM bridge did not become ready: {status}")


def bridge_status(
    timeout_sec: float = 3.0,
    *,
    repo_root: Path | str | None = None,
    route: ModelRoute | None = None,
    builder_route: ModelRoute | None = None,
    judge_route: ModelRoute | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    url = f"http://127.0.0.1:{BRIDGE_PORT}/health/liveliness"
    try:
        response = httpx.get(url, timeout=timeout_sec)
    except httpx.HTTPError as exc:
        return {"ok": False, "url": url, "error": str(exc)}
    ok = response.status_code < 400
    try:
        body: Any = response.json()
    except json.JSONDecodeError:
        body = response.text
    status = {
        "ok": ok,
        "url": url,
        "status_code": response.status_code,
        "body": body,
    }
    if not ok or route is None:
        return status
    try:
        status.update(
            _verify_bridge_runtime(
                route,
                repo_root=repo_root,
                builder_route=builder_route,
                judge_route=judge_route,
                env=env,
            )
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, yaml.YAMLError) as exc:
        return {**status, "ok": False, "error": str(exc)}
    return status


def bridge_runtime_attestation(
    route: ModelRoute,
    *,
    repo_root: Path | str | None = None,
    builder_route: ModelRoute | None = None,
    judge_route: ModelRoute | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    status = bridge_status(
        repo_root=repo_root,
        route=route,
        builder_route=builder_route,
        judge_route=judge_route,
        env=env,
    )
    if not status.get("ok"):
        raise RuntimeError(
            f"bridge runtime does not match the resolved model route: "
            f"{status.get('error') or status.get('body') or status}"
        )
    return {**dict(status["runtime_lock"]), "resolved_image_id": status["resolved_image_id"]}


def _verify_bridge_runtime(
    route: ModelRoute,
    *,
    repo_root: Path | str | None,
    builder_route: ModelRoute | None,
    judge_route: ModelRoute | None,
    env: Mapping[str, str] | None,
) -> dict[str, Any]:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    runtime_dir = root / BRIDGE_RUNTIME_DIR
    expected = bridge_runtime_lock_for_route(
        route,
        builder_route=builder_route,
        judge_route=judge_route,
        env=env,
    )
    lock_path = runtime_dir / BRIDGE_LOCK_NAME
    actual = json.loads(lock_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError(
            "generated bridge runtime lock differs from the selected route"
        )
    config_path = (runtime_dir / BRIDGE_CONFIG_NAME).resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or _json_digest(document) != actual[
        "config_sha256"
    ]:
        raise RuntimeError("bridge config differs from its runtime lock")

    result = subprocess.run(
        ["docker", "inspect", "fugue-litellm-bridge"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "bridge container is unavailable")
    [container] = json.loads(result.stdout)
    configured_image = str((container.get("Config") or {}).get("Image") or "")
    if configured_image != LITELLM_IMAGE:
        raise RuntimeError(
            f"bridge image differs: expected {LITELLM_IMAGE}, found {configured_image}"
        )
    expected_command = docker_compose_for_route(
        route, builder_route=builder_route, judge_route=judge_route
    )["services"]["bridge"]["command"]
    if (container.get("Config") or {}).get("Cmd") != expected_command:
        raise RuntimeError(
            "bridge container command differs from the locked compose plan"
        )
    mounts = container.get("Mounts") or []
    mounted = next(
        (
            item
            for item in mounts
            if str(item.get("Destination") or "") == "/app/config.yaml"
        ),
        None,
    )
    if (
        not mounted
        or Path(str(mounted.get("Source") or "")).resolve() != config_path
        or bool(mounted.get("RW"))
    ):
        raise RuntimeError("bridge container is not using the locked read-only config")
    image_id = str(container.get("Image") or "")
    if not image_id.startswith("sha256:"):
        raise RuntimeError("bridge container has no immutable image id")
    return {"runtime_lock": actual, "resolved_image_id": image_id}


def _json_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
