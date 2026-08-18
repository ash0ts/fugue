from __future__ import annotations

import os
from pathlib import Path

from fugue.bench.executables import resolve_console_script


def test_console_script_resolves_beside_interpreter_when_path_is_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts = tmp_path / "venv" / "bin"
    scripts.mkdir(parents=True)
    python = scripts / "python"
    python.write_text("")
    harbor = scripts / "harbor"
    harbor.write_text("#!/bin/sh\n")
    harbor.chmod(0o755)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("fugue.bench.executables.sys.executable", str(python))
    monkeypatch.setattr("fugue.bench.executables.sysconfig.get_path", lambda _name: None)

    assert resolve_console_script("harbor") == harbor.absolute().as_posix()


def test_console_script_rejects_non_executable_sibling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts = tmp_path / "venv" / "bin"
    scripts.mkdir(parents=True)
    python = scripts / "python"
    python.write_text("")
    harbor = scripts / "harbor"
    harbor.write_text("#!/bin/sh\n")
    harbor.chmod(0o600)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("fugue.bench.executables.sys.executable", str(python))
    monkeypatch.setattr("fugue.bench.executables.sysconfig.get_path", lambda _name: None)

    assert not os.access(harbor, os.X_OK)
    assert resolve_console_script("harbor") is None
