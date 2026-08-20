from __future__ import annotations

import os
from pathlib import Path

from fugue.bench.executables import (
    resolve_console_script,
    resolve_console_script_python,
)
from fugue.bench.execution import _host_process_command
from fugue.doctor import doctor_report


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


def test_console_script_prefers_active_environment_over_path(
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
    path_harbor = tmp_path / "user-bin" / "harbor"
    path_harbor.parent.mkdir()
    path_harbor.write_text("#!/bin/sh\n")
    path_harbor.chmod(0o755)
    monkeypatch.setattr("fugue.bench.executables.sys.executable", str(python))
    monkeypatch.setattr(
        "fugue.bench.executables.sysconfig.get_path", lambda _name: None
    )
    monkeypatch.setattr(
        "fugue.bench.executables.shutil.which",
        lambda name: str(path_harbor) if name == "harbor" else None,
    )

    assert resolve_console_script("harbor") == harbor.absolute().as_posix()
    assert doctor_report(tmp_path)["optional_features"]["local_runner"][
        "executable"
    ] == harbor.absolute().as_posix()
    assert _host_process_command(("harbor", "run"))[0] == (
        harbor.absolute().as_posix()
    )


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


def test_console_script_python_supports_windows_sibling(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    harbor = scripts / "harbor.exe"
    harbor.write_bytes(b"launcher")
    harbor.chmod(0o755)
    python = scripts / "python.exe"
    python.write_bytes(b"interpreter")
    python.chmod(0o755)

    assert resolve_console_script_python(harbor) == python.as_posix()


def test_console_script_python_preserves_path_only_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    active_bin = tmp_path / "active" / "bin"
    active_bin.mkdir(parents=True)
    active_python = active_bin / "python"
    active_python.write_bytes(b"active")
    active_python.chmod(0o755)
    tool_bin = tmp_path / "path-tool" / "bin"
    tool_bin.mkdir(parents=True)
    harbor = tool_bin / "harbor"
    harbor.write_text("#!/not/used\n")
    harbor.chmod(0o755)
    tool_python = tool_bin / "python"
    tool_python.write_bytes(b"tool")
    tool_python.chmod(0o755)
    monkeypatch.setattr(
        "fugue.bench.executables.sys.executable", active_python.as_posix()
    )
    monkeypatch.setattr(
        "fugue.bench.executables.sysconfig.get_path", lambda _name: None
    )
    monkeypatch.setattr(
        "fugue.bench.executables.shutil.which",
        lambda name: harbor.as_posix() if name == "harbor" else None,
    )

    selected = resolve_console_script("harbor")

    assert selected == harbor.as_posix()
    assert resolve_console_script_python(selected) == tool_python.as_posix()


def test_console_script_python_prefers_direct_shebang_over_path_sibling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    active_bin = tmp_path / "active" / "bin"
    active_bin.mkdir(parents=True)
    active_python = active_bin / "python"
    active_python.write_bytes(b"active")
    active_python.chmod(0o755)
    actual_bin = tmp_path / "actual-tool" / "bin"
    actual_bin.mkdir(parents=True)
    actual_python = actual_bin / "python"
    actual_python.write_bytes(b"actual")
    actual_python.chmod(0o755)
    path_bin = tmp_path / "path" / "bin"
    path_bin.mkdir(parents=True)
    harbor = path_bin / "harbor"
    harbor.write_text(f"#!{actual_python}\n")
    harbor.chmod(0o755)
    unrelated_sibling = path_bin / "python"
    unrelated_sibling.write_bytes(b"unrelated")
    unrelated_sibling.chmod(0o755)
    monkeypatch.setattr(
        "fugue.bench.executables.sys.executable", active_python.as_posix()
    )
    monkeypatch.setattr(
        "fugue.bench.executables.sysconfig.get_path", lambda _name: None
    )

    assert resolve_console_script_python(harbor) == actual_python.as_posix()
