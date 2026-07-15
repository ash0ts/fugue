"""Protocol gateway for content-addressed Fugue candidate deployments."""

from typing import Any

__all__ = ["create_app"]


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Load the optional FastAPI gateway only when serving is requested."""
    from fugue.serve.app import create_app as factory

    return factory(*args, **kwargs)
