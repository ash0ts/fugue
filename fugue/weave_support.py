from __future__ import annotations

import base64
import os
from collections.abc import Awaitable, Callable, Mapping
from threading import Lock
from typing import Any

_INITIALIZED_PROJECTS: set[str] = set()
_LOCK = Lock()

WEAVE_AGENTS_BASE_URL = "https://trace.wandb.ai"
WEAVE_AGENTS_OTEL_ENDPOINT = f"{WEAVE_AGENTS_BASE_URL}/agents/otel/v1/traces"


_WEAVE_ENV_KEYS = (
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
    "WANDB_PUBLIC_BASE_URL",
    "WF_TRACE_SERVER_URL",
    "WEAVE_INSECURE_DISABLE_SSL",
)


def _apply_weave_environment(env: Mapping[str, str] | None) -> None:
    if env is None:
        return
    for key in _WEAVE_ENV_KEYS:
        value = env.get(key)
        if value is not None:
            os.environ[key] = value


def initialize_weave(project: str, env: Mapping[str, str] | None = None) -> Any:
    try:
        import weave
    except ImportError as exc:
        raise RuntimeError("weave is not installed") from exc
    with _LOCK:
        _apply_weave_environment(env)
        if project not in _INITIALIZED_PROJECTS:
            weave.init(project)
            _INITIALIZED_PROJECTS.add(project)
    return weave


def weave_agents_otel_headers(project: str, api_key: str) -> str:
    """Return OTLP env headers without writing credentials to a config file."""
    token = base64.b64encode(f"api:{api_key}".encode()).decode()
    return f"project_id={project},Authorization=Basic%20{token}"


async def trace_async_operation(
    name: str,
    metadata: dict[str, Any],
    env: dict[str, str],
    operation: Callable[[], Awaitable[Any]],
    summarize: Callable[[Any], Any],
) -> Any:
    if not env.get("WANDB_API_KEY", "").strip():
        return await operation()
    from fugue.model_plane import trace_project_slug

    try:
        weave = initialize_weave(trace_project_slug(env), env)
    except RuntimeError:
        return await operation()
    result: Any = None

    @weave.op(name=name)
    async def traced(inputs: dict[str, Any]) -> Any:
        nonlocal result
        result = await operation()
        return summarize(result)

    await traced(metadata)
    return result
