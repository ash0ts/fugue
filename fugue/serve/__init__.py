"""Protocol gateway for content-addressed Fugue candidate deployments."""

import sys
from typing import Any

__all__ = ["create_app"]


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Load the optional FastAPI gateway only when serving is requested."""
    if sys.version_info < (3, 13):
        raise RuntimeError("candidate serving requires Python 3.13 or newer")
    from fugue.serve.app import create_app as factory

    return factory(*args, **kwargs)
