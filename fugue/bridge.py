from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from fugue.model_plane import (
    BRIDGE_MASTER_KEY_ENV,
    ModelRoute,
    bridge_master_key,
    resolve_model_route,
)

BRIDGE_PORT = 4000
BRIDGE_RUNTIME_DIR = Path(".fugue") / "bridge"
BRIDGE_CONFIG_NAME = "litellm.config.yaml"
BRIDGE_COMPOSE_NAME = "docker-compose.yaml"


@dataclass(frozen=True)
class BridgeFiles:
    runtime_dir: Path
    config_path: Path
    compose_path: Path


def litellm_config_for_route(route: ModelRoute) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": route.litellm_model,
        "api_key": f"os.environ/{route.api_key_env}",
    }
    if route.provider in {"wandb", "openai"} and route.chat_base_url:
        params["api_base"] = route.chat_base_url
    if route.provider == "anthropic" and route.messages_base_url:
        params["api_base"] = route.messages_base_url
    return {
        "model_list": [
            {
                "model_name": "*",
                "litellm_params": params,
            }
        ],
        "litellm_settings": {
            "drop_params": True,
        },
        "general_settings": {
            "master_key": f"os.environ/{BRIDGE_MASTER_KEY_ENV}",
        },
    }


def docker_compose_for_route(route: ModelRoute) -> dict[str, Any]:
    return {
        "services": {
            "bridge": {
                "image": "ghcr.io/berriai/litellm:main-latest",
                "container_name": "fugue-litellm-bridge",
                "ports": [f"127.0.0.1:{BRIDGE_PORT}:4000"],
                "volumes": [f"./{BRIDGE_CONFIG_NAME}:/app/config.yaml:ro"],
                "environment": {
                    route.api_key_env: f"${{{route.api_key_env}}}",
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


def write_bridge_files(
    route: ModelRoute, repo_root: Path | str | None = None
) -> BridgeFiles:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    runtime_dir = root / BRIDGE_RUNTIME_DIR
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_path = runtime_dir / BRIDGE_CONFIG_NAME
    compose_path = runtime_dir / BRIDGE_COMPOSE_NAME
    config_path.write_text(
        yaml.safe_dump(litellm_config_for_route(route), sort_keys=False)
    )
    compose_path.write_text(
        yaml.safe_dump(docker_compose_for_route(route), sort_keys=False)
    )
    return BridgeFiles(
        runtime_dir=runtime_dir,
        config_path=config_path,
        compose_path=compose_path,
    )


def bridge_up(
    model: str | None = None,
    *,
    repo_root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> BridgeFiles:
    route = resolve_model_route(model, env=env)
    files = write_bridge_files(route, repo_root)
    subprocess.run(
        ["docker", "compose", "-f", files.compose_path.as_posix(), "up", "-d"],
        cwd=Path.cwd() if repo_root is None else Path(repo_root),
        check=True,
        env=dict(env) if env is not None else None,
    )
    return files


def bridge_status(timeout_sec: float = 3.0) -> dict[str, Any]:
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
    return {
        "ok": ok,
        "url": url,
        "status_code": response.status_code,
        "body": body,
    }


def bridge_env_for_route(route: ModelRoute, env: Mapping[str, str]) -> dict[str, str]:
    values = dict(env)
    values.setdefault(BRIDGE_MASTER_KEY_ENV, bridge_master_key(values))
    return values
