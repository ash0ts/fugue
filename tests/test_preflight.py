from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fugue.preflight import (
    HARBOR_VERSION,
    PreflightCheck,
    harbor_import_check,
    run_preflight,
    validate_harbor_job_configs,
)
from fugue.weave_support import resolved_weave_trace_server_url


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
    assert commands == [[harbor_python.as_posix(), "-c", "import fugue.agents"]]


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


def test_preflight_reports_the_resolved_provider_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: None)

    checks = run_preflight(
        "anthropic/claude-haiku-4-5-20251001",
        repo_root=tmp_path,
        env={"ANTHROPIC_API_KEY": "present"},
        live=False,
    )

    route = next(check for check in checks if check.name == "model route")
    assert route.detail == (
        "anthropic/claude-haiku-4-5-20251001 via anthropic at https://api.anthropic.com"
    )


def test_job_configs_are_validated_by_harbor_tool_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool_bin = tmp_path / "harbor" / "bin"
    tool_bin.mkdir(parents=True)
    harbor = tool_bin / "harbor"
    harbor.touch()
    harbor_python = tool_bin / "python"
    harbor_python.touch()
    config = tmp_path / "job.json"
    config.write_text("{}")
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: str(harbor))
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("fugue.preflight.subprocess.run", fake_run)

    validate_harbor_job_configs([config])

    [command] = commands
    assert command[0] == harbor_python.as_posix()
    assert command[1] == "-c"
    assert command[-2:] == [HARBOR_VERSION, config.as_posix()]


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


def test_preflight_skips_bridge_when_selected_harnesses_are_provider_direct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: None)

    def unexpected_bridge_status(**kwargs):
        raise AssertionError(f"bridge status must not run: {kwargs}")

    monkeypatch.setattr("fugue.preflight.bridge_status", unexpected_bridge_status)

    checks = run_preflight(
        "wandb/zai-org/GLM-5.2",
        repo_root=tmp_path,
        env={},
        live=True,
        harnesses=("hermes", "openclaw"),
    )

    bridge = next(check for check in checks if check.name == "bridge health")
    assert bridge.ok is True
    assert bridge.detail.startswith("not required")


def test_preflight_attests_bridge_for_selected_native_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: None)
    captured = {}

    def fake_bridge_status(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "runtime_lock": {"image": "bridge@example"},
            "resolved_image_id": "sha256:resolved",
        }

    monkeypatch.setattr("fugue.preflight.bridge_status", fake_bridge_status)

    checks = run_preflight(
        "wandb/zai-org/GLM-5.2",
        repo_root=tmp_path,
        env={},
        live=True,
        harnesses=("codex",),
    )

    bridge = next(check for check in checks if check.name == "bridge health")
    assert bridge.ok is True
    assert bridge.detail == "locked bridge@example as sha256:resolved"
    assert captured["repo_root"] == tmp_path
    assert captured["route"].display_model == "wandb/zai-org/GLM-5.2"


def test_wba_preflight_only_requires_bridge_for_proxy_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: None)

    def unexpected_bridge_status(**kwargs):
        raise AssertionError(f"bridge status must not run: {kwargs}")

    monkeypatch.setattr("fugue.preflight.bridge_status", unexpected_bridge_status)
    checks = run_preflight(
        "wandb/zai-org/GLM-5.2",
        repo_root=tmp_path,
        env={},
        live=True,
        harnesses=("wba-responses",),
        wba_transport_profiles=("responses-inline", "chat-inline"),
    )
    assert next(check for check in checks if check.name == "bridge health").ok


def test_wba_proxy_profile_attests_bridge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "fugue.preflight.bridge_status",
        lambda **kwargs: {
            "ok": True,
            "runtime_lock": {"image": "bridge@example"},
            "resolved_image_id": "sha256:resolved",
        },
    )
    checks = run_preflight(
        "wandb/zai-org/GLM-5.2",
        repo_root=tmp_path,
        env={},
        live=True,
        harnesses=("wba-responses",),
        wba_transport_profiles=("responses-proxy", "responses-inline"),
    )
    assert next(check for check in checks if check.name == "bridge health").ok


def test_wandb_preflight_attributes_inference_to_trace_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_headers: dict[str, str] = {}

    def fake_get(url, *, headers, timeout):
        del url, timeout
        captured_headers.update(headers)
        return type("Response", (), {"status_code": 200})()

    monkeypatch.setattr("fugue.preflight.httpx.get", fake_get)
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "fugue.preflight.bridge_status", lambda: {"ok": False, "error": "offline"}
    )

    checks = run_preflight(
        "wandb/zai-org/GLM-5.2",
        repo_root=tmp_path,
        env={
            "WANDB_API_KEY": "test-only",
            "WANDB_ENTITY": "team",
            "WANDB_PROJECT": "billing-project",
        },
        live=True,
    )

    assert next(check for check in checks if check.name == "provider metadata").ok
    assert captured_headers["OpenAI-Project"] == "team/billing-project"


def test_weave_endpoint_resolution_prefers_run_environment() -> None:
    assert (
        resolved_weave_trace_server_url(
            {
                "WANDB_BASE_URL": "https://api.wandb.ai",
                "WANDB_PUBLIC_BASE_URL": "https://ignored.example",
                "WF_TRACE_SERVER_URL": "https://trace.wandb.ai/",
            }
        )
        == "https://trace.wandb.ai"
    )


def test_live_preflight_probes_the_resolved_weave_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    urls: list[str] = []

    def fake_get(url, *, headers, timeout):
        del headers, timeout
        urls.append(url)
        return type("Response", (), {"status_code": 200})()

    monkeypatch.setattr("fugue.preflight.httpx.get", fake_get)
    monkeypatch.setattr("fugue.preflight.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "fugue.preflight.bridge_status", lambda: {"ok": False, "error": "offline"}
    )

    checks = run_preflight(
        "wandb/zai-org/GLM-5.2",
        repo_root=tmp_path,
        env={
            "WANDB_API_KEY": "test-only",
            "WANDB_BASE_URL": "https://api.wandb.ai",
            "WF_TRACE_SERVER_URL": "https://trace.wandb.ai",
        },
        live=True,
    )

    assert next(check for check in checks if check.name == "weave endpoint").ok
    assert "https://trace.wandb.ai/server_info" in urls
