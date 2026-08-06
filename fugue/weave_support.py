from __future__ import annotations

import base64
import os
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from threading import RLock
from typing import Any

from fugue.model_plane import (
    DEFAULT_WEAVE_TRACE_BASE_URL,
    resolve_evidence_destination,
    trace_api_key,
    trace_project_environment,
)

_ACTIVE_DESTINATION_DIGEST: str | None = None
# Weave's active client and the endpoint variables used to create it are
# process-global.  A plain init lock is insufficient: another Study can switch
# the client after ``initialize_weave`` returns but before the caller creates a
# Dataset, Evaluation, or prediction.  Callers that perform remote work use the
# destination scope below so activation and the dependent SDK operation are one
# atomic critical section.  RLock keeps nested lifecycle helpers safe.
_LOCK = RLock()
_WEAVE_CLIENT_CONTEXT_UNAVAILABLE = object()
WEAVE_AGENTS_BASE_URL = DEFAULT_WEAVE_TRACE_BASE_URL

_WEAVE_ENV_KEYS = (
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
    "WANDB_APP_BASE_URL",
    "WANDB_PUBLIC_BASE_URL",
    "WF_TRACE_SERVER_URL",
    "FUGUE_WEAVE_BASE_URL",
    "FUGUE_WEAVE_TRACE_SERVER_URL",
    "FUGUE_EVIDENCE_DESTINATION_DIGEST",
    "FUGUE_EVIDENCE_DESTINATION_JSON",
    "WEAVE_INSECURE_DISABLE_SSL",
)


def _apply_weave_environment(env: Mapping[str, str] | None) -> None:
    if env is None:
        return
    trace_key = trace_api_key(env)
    if trace_key:
        os.environ["WANDB_API_KEY"] = trace_key
    for key in _WEAVE_ENV_KEYS:
        value = env.get(key)
        if value is not None:
            os.environ[key] = value


def _active_weave_project_slug() -> str | None | object:
    """Return the active SDK project, or a sentinel when context introspection is absent."""

    try:
        from weave.trace.context.weave_client_context import get_weave_client
    except (ImportError, ModuleNotFoundError):
        return _WEAVE_CLIENT_CONTEXT_UNAVAILABLE
    client = get_weave_client()
    if client is None:
        return None
    project = str(getattr(client, "project", "") or "")
    entity = str(getattr(client, "entity", "") or "")
    if not project:
        return None
    if "/" in project or not entity:
        return project
    return f"{entity}/{project}"


def _initialize_weave_locked(
    project: str, env: Mapping[str, str] | None = None
) -> Any:
    try:
        import weave
    except ImportError as exc:
        raise RuntimeError("weave is not installed") from exc
    bound_env = trace_project_environment(project, env)
    destination = resolve_evidence_destination(bound_env)
    global _ACTIVE_DESTINATION_DIGEST
    _apply_weave_environment(bound_env)
    active_project = _active_weave_project_slug()
    destination_changed = destination.destination_digest != _ACTIVE_DESTINATION_DIGEST
    client_missing_or_wrong = (
        active_project is not _WEAVE_CLIENT_CONTEXT_UNAVAILABLE
        and active_project != destination.project_slug
    )
    if destination_changed or client_missing_or_wrong:
        weave.init(destination.project_slug)
        _ACTIVE_DESTINATION_DIGEST = destination.destination_digest
    return weave


@contextmanager
def activated_weave_destination(
    project: str, env: Mapping[str, str] | None = None
) -> Any:
    """Hold the exact Weave destination for one complete SDK operation.

    The scope deliberately serializes only host-side evidence operations. Agent
    containers still run concurrently. This prevents concurrent Studies from
    observing a client or endpoint activated by a different destination.
    """

    with _LOCK:
        yield _initialize_weave_locked(project, env)


@contextmanager
def serialized_weave_destination() -> Any:
    """Serialize a caller-controlled activate-and-publish SDK operation."""

    with _LOCK:
        yield


def initialize_weave(project: str, env: Mapping[str, str] | None = None) -> Any:
    """Activate the exact evidence destination, including A → B → A switches.

    This compatibility helper protects activation itself. Multi-step SDK work
    must use :func:`activated_weave_destination` so the active client cannot be
    switched between activation and publication.
    """

    with activated_weave_destination(project, env) as weave:
        return weave


def weave_agents_otel_headers(project: str, api_key: str) -> str:
    """Return OTLP env headers without writing credentials to a config file."""
    token = base64.b64encode(f"api:{api_key}".encode()).decode()
    return f"project_id={project},Authorization=Basic%20{token}"


def resolved_weave_trace_server_url(env: Mapping[str, str]) -> str:
    """Resolve the same trace endpoint as the pinned Weave SDK without mutating env."""

    return resolve_evidence_destination(env).trace_base_url


def weave_agents_otel_endpoint(env: Mapping[str, str]) -> str:
    """Resolve the Agent OTLP endpoint from the evidence destination."""

    trace_server = resolved_weave_trace_server_url(env)
    return f"{trace_server.rstrip('/')}/agents/otel/v1/traces"


async def trace_async_operation(
    name: str,
    metadata: dict[str, Any],
    env: dict[str, str],
    operation: Callable[[], Awaitable[Any]],
    summarize: Callable[[Any], Any],
) -> Any:
    if not trace_api_key(env):
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
