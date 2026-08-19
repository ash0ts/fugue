from __future__ import annotations

import base64
import os
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from threading import RLock
from typing import Any

from fugue.model_plane import (
    DEFAULT_WEAVE_TRACE_BASE_URL,
    EvidenceDestinationV1,
    evidence_destination_environment,
    resolve_evidence_destination,
    trace_api_key,
    trace_project_environment,
)

_ACTIVE_DESTINATION_DIGEST: str | None = None
_ACTIVE_DESTINATION_LEASE_DIGEST: str | None = None
_ACTIVE_DESTINATION_LEASE_COUNT = 0
# W&B and Weave SDKs route through process-global environment and client state.
# Every adapter that changes that state must hold this lock through readback.
EVIDENCE_ROUTING_LOCK = RLock()
_WEAVE_CLIENT_CONTEXT_UNAVAILABLE = object()
WEAVE_AGENTS_BASE_URL = DEFAULT_WEAVE_TRACE_BASE_URL

_WEAVE_ENV_KEYS = (
    "FUGUE_WEAVE_API_KEY",
    "FUGUE_WEAVE_PROJECT",
    "WEAVE_PROJECT",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
    "WANDB_APP_BASE_URL",
    "WANDB_PUBLIC_BASE_URL",
    "WF_TRACE_SERVER_URL",
    "FUGUE_WEAVE_BASE_URL",
    "FUGUE_WEAVE_TRACE_SERVER_URL",
    "FUGUE_EVIDENCE_DESTINATION_DIGEST",
    "FUGUE_EVIDENCE_DESTINATION_JSON",
    "WANDB_INSECURE_DISABLE_SSL",
    "WEAVE_INSECURE_DISABLE_SSL",
)
_WEAVE_CONTEXT_ONLY_ENV_KEYS = frozenset(
    {
        "FUGUE_WEAVE_API_KEY",
        "FUGUE_WEAVE_PROJECT",
        "WEAVE_PROJECT",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_API_KEY",
    }
)

def _apply_weave_environment(env: Mapping[str, str] | None) -> None:
    for key in _WEAVE_ENV_KEYS:
        os.environ.pop(key, None)
    if env is None:
        return
    for key in _WEAVE_ENV_KEYS:
        if key in _WEAVE_CONTEXT_ONLY_ENV_KEYS:
            continue
        value = env.get(key)
        if value is not None:
            os.environ[key] = value
    trace_key = trace_api_key(env)
    if trace_key:
        os.environ["WANDB_API_KEY"] = trace_key


def _active_weave_project_slug(weave: Any) -> str | None | object:
    """Return the active SDK project through Weave's public client API."""

    get_client = getattr(weave, "get_client", None)
    if not callable(get_client):
        return _WEAVE_CLIENT_CONTEXT_UNAVAILABLE
    client = get_client()
    if client is None:
        return None
    project = str(getattr(client, "project", "") or "")
    entity = str(getattr(client, "entity", "") or "")
    if not project:
        return None
    if "/" in project or not entity:
        return project
    return f"{entity}/{project}"


def _activate_weave_locked(
    weave: Any,
    project: str | EvidenceDestinationV1,
    env: Mapping[str, str] | None,
) -> None:
    """Activate one destination while the caller holds the routing lock."""

    if isinstance(project, EvidenceDestinationV1):
        observed = resolve_evidence_destination(
            trace_project_environment(project.project_slug, env)
        )
        if observed != project:
            raise ValueError(
                "Weave environment disagrees with the immutable evidence destination"
            )
        bound_env = evidence_destination_environment(project, env)
        destination = resolve_evidence_destination(bound_env)
    else:
        bound_env = trace_project_environment(project, env)
        destination = resolve_evidence_destination(bound_env)
    global _ACTIVE_DESTINATION_DIGEST
    if (
        _ACTIVE_DESTINATION_LEASE_DIGEST is not None
        and destination.destination_digest != _ACTIVE_DESTINATION_LEASE_DIGEST
    ):
        raise RuntimeError(
            "cannot switch Weave destination while a live Evaluation lease is active"
        )
    _apply_weave_environment(bound_env)
    active_project = _active_weave_project_slug(weave)
    destination_changed = destination.destination_digest != _ACTIVE_DESTINATION_DIGEST
    client_missing_or_wrong = (
        active_project is not _WEAVE_CLIENT_CONTEXT_UNAVAILABLE
        and active_project != destination.project_slug
    )
    if destination_changed or client_missing_or_wrong:
        weave.init(destination.project_slug)
        _ACTIVE_DESTINATION_DIGEST = destination.destination_digest


@contextmanager
def weave_destination_session(
    project: str | EvidenceDestinationV1,
    env: Mapping[str, str] | None = None,
) -> Iterator[Any]:
    """Hold the process-global Weave destination for one complete operation.

    Weave selects its client and endpoints through process-global state. Holding
    this session prevents another thread from switching destinations between a
    publish call and its authoritative readback. Environment values are restored
    when the operation completes.
    """

    try:
        import weave
    except ImportError as exc:
        raise RuntimeError("weave is not installed") from exc
    with EVIDENCE_ROUTING_LOCK:
        previous = {key: os.environ.get(key) for key in _WEAVE_ENV_KEYS}
        try:
            _activate_weave_locked(weave, project, env)
            yield weave
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def initialize_weave(project: str, env: Mapping[str, str] | None = None) -> Any:
    """Activate the exact evidence destination, including A → B → A switches."""

    try:
        import weave
    except ImportError as exc:
        raise RuntimeError("weave is not installed") from exc
    with EVIDENCE_ROUTING_LOCK:
        _activate_weave_locked(weave, project, env)
    return weave


def acquire_weave_destination_lease(
    project: str,
    env: Mapping[str, str] | None = None,
) -> str:
    """Hold one destination while live Evaluation roots remain open."""

    try:
        import weave
    except ImportError as exc:
        raise RuntimeError("weave is not installed") from exc
    with EVIDENCE_ROUTING_LOCK:
        bound_env = trace_project_environment(project, env)
        destination = resolve_evidence_destination(bound_env)
        global _ACTIVE_DESTINATION_LEASE_DIGEST, _ACTIVE_DESTINATION_LEASE_COUNT
        if (
            _ACTIVE_DESTINATION_LEASE_DIGEST is not None
            and _ACTIVE_DESTINATION_LEASE_DIGEST != destination.destination_digest
        ):
            raise RuntimeError(
                "another live Evaluation owns the process-global Weave destination"
            )
        _activate_weave_locked(weave, project, env)
        _ACTIVE_DESTINATION_LEASE_DIGEST = destination.destination_digest
        _ACTIVE_DESTINATION_LEASE_COUNT += 1
        return destination.destination_digest


def release_weave_destination_lease(lease_digest: str) -> None:
    """Release a live Evaluation destination lease after roots close."""

    with EVIDENCE_ROUTING_LOCK:
        global _ACTIVE_DESTINATION_LEASE_DIGEST, _ACTIVE_DESTINATION_LEASE_COUNT
        if (
            not lease_digest
            or _ACTIVE_DESTINATION_LEASE_DIGEST != lease_digest
            or _ACTIVE_DESTINATION_LEASE_COUNT <= 0
        ):
            raise RuntimeError("live Weave destination lease identity disagrees")
        _ACTIVE_DESTINATION_LEASE_COUNT -= 1
        if _ACTIVE_DESTINATION_LEASE_COUNT == 0:
            _ACTIVE_DESTINATION_LEASE_DIGEST = None


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
    if str(env.get("FUGUE_EVIDENCE_MODE") or "").strip() == "local":
        return await operation()
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
