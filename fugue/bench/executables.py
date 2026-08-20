from __future__ import annotations

import os
import shlex
import shutil
import sys
import sysconfig
from pathlib import Path


def resolve_console_script(name: str) -> str | None:
    """Resolve an installed console script without assuming its bin dir is on PATH.

    Calling an absolute ``fugue`` script does not activate its virtual environment or
    prepend the environment's scripts directory to ``PATH``. Optional executors such
    as Harbor are nevertheless installed beside Fugue. Prefer those trusted scripts
    before consulting ``PATH`` so a user-level tool cannot replace the executor that
    was installed with the running Fugue distribution.
    """

    candidates = [Path(sys.executable).parent / name]
    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidates.append(Path(scripts) / name)
    if os.name == "nt":
        candidates.extend(path.with_suffix(".exe") for path in tuple(candidates))

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.absolute().as_posix()

    return shutil.which(name)


def resolve_console_script_python(executable: str | Path) -> str | None:
    """Resolve the Python interpreter that owns a selected console script.

    A script selected from the running Fugue environment uses ``sys.executable``.
    A PATH-only fallback uses an executable sibling or a direct absolute shebang.
    The latter deliberately rejects ``env`` and shell indirection.
    """

    selected = Path(executable).absolute()
    environment_dirs = {Path(sys.executable).absolute().parent}
    if scripts := sysconfig.get_path("scripts"):
        environment_dirs.add(Path(scripts).absolute())
    if selected.parent in environment_dirs:
        active = Path(sys.executable).absolute()
        if _is_executable_file(active):
            return active.as_posix()

    if selected.suffix.lower() != ".exe":
        shebang_python = _direct_absolute_shebang_python(selected)
        if shebang_python is not None:
            return shebang_python

    selected_parent = selected.resolve().parent
    names = (
        ("python.exe", "python")
        if selected.suffix.lower() == ".exe" or os.name == "nt"
        else ("python", "python.exe")
    )
    for name in names:
        candidate = selected_parent / name
        if _is_executable_file(candidate):
            return candidate.absolute().as_posix()

    return None


def _direct_absolute_shebang_python(selected: Path) -> str | None:
    """Return one direct absolute shebang interpreter without shell expansion."""

    try:
        first_line = selected.read_text(encoding="utf-8").splitlines()[0]
    except (IndexError, OSError, UnicodeError):
        return None
    if not first_line.startswith("#!"):
        return None
    try:
        command = shlex.split(first_line.removeprefix("#!").strip())
    except ValueError:
        return None
    if len(command) != 1:
        return None
    interpreter = Path(command[0])
    if not interpreter.is_absolute() or not _is_executable_file(interpreter):
        return None
    return interpreter.as_posix()


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)
