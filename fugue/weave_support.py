from __future__ import annotations

from collections.abc import Awaitable, Callable
from threading import Lock
from typing import Any

_INITIALIZED_PROJECTS: set[str] = set()
_LOCK = Lock()


def initialize_weave(project: str) -> Any:
    try:
        import weave
    except ImportError as exc:
        raise RuntimeError("weave is not installed") from exc
    with _LOCK:
        if project not in _INITIALIZED_PROJECTS:
            weave.init(project)
            _INITIALIZED_PROJECTS.add(project)
    return weave


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
        weave = initialize_weave(trace_project_slug(env))
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
