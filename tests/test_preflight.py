from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fugue.preflight import PreflightCheck, harbor_import_check, run_preflight


def test_harbor_import_check_uses_resolved_tool_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool_bin = tmp_path / "tools" / "harbor" / "bin"
    tool_bin.mkdir(parents=True)
    harbor = tool_bin / "harbor"
    harbor.touch()
    harbor_python = tool_bin / "python"
    harbor_python.touch()

    launcher_dir = tmp_path / "bin"
    launcher_dir.mkdir()
    launcher = launcher_dir / "harbor"
    launcher.symlink_to(harbor)
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: str(launcher))

    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("fugue.preflight.subprocess.run", fake_run)

    check = harbor_import_check(tmp_path)

    assert check.ok is True
    assert check.name == "adapters"
    assert commands == [
        [harbor_python.as_posix(), "-c", "import fugue.agents"]
    ]


def test_preflight_reports_harbor_runtime_adapter_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        "fugue.preflight.harbor_import_check",
        lambda root: PreflightCheck("adapters", True, "harbor runtime"),
    )

    checks = run_preflight(
        "wandb/zai-org/GLM-5.2",
        repo_root=tmp_path,
        env={"WANDB_API_KEY": "present"},
        live=False,
    )

    adapters = next(check for check in checks if check.name == "adapters")
    assert adapters.ok is True
    assert adapters.detail == "harbor runtime"


def test_live_preflight_is_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "fugue.preflight.bridge_status",
        lambda: {"ok": False, "error": "offline"},
    )

    checks = run_preflight(
        "openai/gpt-5",
        repo_root=tmp_path,
        env={},
        live=True,
    )

    assert next(check for check in checks if check.name == "bridge health").ok is False
    assert not (tmp_path / ".fugue").exists()
