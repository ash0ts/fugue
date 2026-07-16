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
DEFAULT_WANDB_BASE_URL = "https://api.wandb.ai"


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


def resolved_weave_trace_server_url(env: Mapping[str, str]) -> str:
    """Resolve the same trace endpoint as the pinned Weave SDK without mutating env."""

    explicit = env.get("WF_TRACE_SERVER_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    public = env.get("WANDB_PUBLIC_BASE_URL", "").strip()
    if public:
        base_url = public.rstrip("/")
    else:
        configured = env.get("WANDB_BASE_URL", "").strip()
        if configured:
            base_url = configured.rstrip("/")
        else:
            try:
                from weave.trace.env import Settings

                base_url = Settings().base_url.rstrip("/")
            except ImportError:
                base_url = DEFAULT_WANDB_BASE_URL
    return (
        WEAVE_AGENTS_BASE_URL
        if base_url == DEFAULT_WANDB_BASE_URL
        else f"{base_url}/traces"
    )


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
    except Exception:
        return await operation()
    sentinel = object()
    result: Any = sentinel
    operation_error: BaseException | None = None
    started = False

    async def traced_operation(inputs: dict[str, Any]) -> Any:
        nonlocal result, operation_error, started
        started = True
        try:
            result = await operation()
        except BaseException as exc:
            operation_error = exc
            raise
        return summarize(result)

    try:
        traced = weave.op(name=name)(traced_operation)
    except Exception:
        return await operation()

    try:
        await traced(metadata)
    except BaseException:
        if operation_error is not None:
            raise operation_error from None
        if not started:
            return await operation()
        if result is not sentinel:
            return result
        raise
    return result
