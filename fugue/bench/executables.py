from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path


def resolve_console_script(name: str) -> str | None:
    """Resolve an installed console script without assuming its bin dir is on PATH.

    Calling an absolute ``fugue`` script does not activate its virtual environment or
    prepend the environment's scripts directory to ``PATH``.  Optional executors such
    as Harbor are nevertheless installed beside Fugue, so look in the interpreter's
    scripts directories after the normal PATH lookup.
    """

    discovered = shutil.which(name)
    if discovered is not None:
        return discovered

    candidates = [Path(sys.executable).parent / name]
    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidates.append(Path(scripts) / name)
    if os.name == "nt":
        candidates.extend(path.with_suffix(".exe") for path in tuple(candidates))

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.absolute().as_posix()
    return None
