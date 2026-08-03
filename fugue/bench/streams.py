from __future__ import annotations

import errno
import sys
from typing import TextIO

_CLOSED_OUTPUT_ERRNOS = frozenset({errno.EBADF, errno.EIO, errno.EPIPE})


def write_stdout_best_effort(
    text: str,
    *,
    stream: TextIO | None = None,
) -> bool:
    """Mirror output without making a caller's terminal part of run correctness."""

    output = sys.stdout if stream is None else stream
    try:
        output.write(text)
        output.flush()
    except ValueError:
        if getattr(output, "closed", False):
            return False
        raise
    except OSError as exc:
        if exc.errno in _CLOSED_OUTPUT_ERRNOS:
            return False
        raise
    return True
