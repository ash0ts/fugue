from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from fugue.bench.cli import main
from fugue.doctor import doctor_report


def test_doctor_is_ready_in_an_empty_workspace(tmp_path: Path) -> None:
    report = doctor_report(tmp_path)

    assert report["ok"] is True
    assert report["readiness"]["mode"] == "observational"
    assert report["readiness"]["requested_capabilities"] == []
    assert report["distribution"]["kind"] == "installed_distribution"
    assert report["workspace"]["kind"] == "unversioned"
    assert report["workspace"]["files"] == 0
    assert report["assets"]["vendor_archive"] is True
    assert "none" in report["assets"]["context_systems"]
    assert report["assets"]["schemas"] > 0
    assert report["assets"]["runtime_groups"]["fugue-context"] == {
        "files": 2,
        "available": True,
    }
    assert report["host"]["architecture"]
    assert report["host"]["free_disk_bytes"] > 0
    assert "daemon_available" in report["host"]["docker"]
    assert report["model_route"] == {
        "selected": False,
        "model": None,
        "provider": None,
        "credential_name": None,
        "credential_present": None,
        "missing_credentials": [],
    }
    assert all(
        item["available"]
        for item in report["assets"]["runtime_groups"].values()
    )


def test_doctor_does_not_require_optional_features(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def missing_version(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr("fugue.doctor.version", missing_version)
    monkeypatch.setattr("fugue.doctor.importlib.util.find_spec", lambda name: None)

    report = doctor_report(tmp_path)

    assert report["ok"] is True
    assert report["optional_features"]["weave"]["installed"] is False
    assert report["optional_features"]["local_runner"]["installed"] is False


def test_required_local_runner_fails_closed_on_missing_runtime_and_credential(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def missing_version(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr("fugue.doctor.version", missing_version)
    monkeypatch.setattr("fugue.doctor.importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(
        "fugue.doctor._docker_status",
        lambda **_kwargs: {
            "cli_available": False,
            "daemon_available": False,
            "detail": "docker CLI not found",
            "network_ready": False,
            "network_detail": "docker CLI not found",
        },
    )

    report = doctor_report(
        tmp_path,
        model="anthropic/claude-sonnet-5",
        env={},
        required_capabilities=("local-runner",),
    )

    assert report["ok"] is False
    assert report["readiness"]["mode"] == "required"
    assert report["readiness"]["requested_capabilities"] == ["local-runner"]
    requirements = report["readiness"]["requirements"]
    assert requirements["harbor_installed"]["ready"] is False
    assert requirements["docker_cli"]["ready"] is False
    assert requirements["docker_daemon"]["ready"] is False
    assert requirements["model_credential"]["ready"] is False
    assert "ANTHROPIC_API_KEY" in requirements["model_credential"]["detail"]


def test_required_local_runner_reports_ready_when_every_gate_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("fugue.doctor._python_version_info", lambda: (3, 13, 0))
    monkeypatch.setattr("fugue.doctor.version", lambda name: "0.18.0")
    monkeypatch.setattr("fugue.doctor.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "fugue.doctor.resolve_console_script", lambda name: f"/venv/bin/{name}"
    )
    monkeypatch.setattr(
        "fugue.doctor._docker_status",
        lambda **_kwargs: {
            "cli_available": True,
            "daemon_available": True,
            "detail": "27.5.1",
            "network_ready": True,
            "network_detail": "probe passed",
        },
    )

    report = doctor_report(
        tmp_path,
        model="anthropic/claude-sonnet-5",
        env={"ANTHROPIC_API_KEY": "present-for-test"},
        required_capabilities=("local-runner",),
    )

    assert report["ok"] is True
    assert all(
        item["ready"]
        for item in report["readiness"]["requirements"].values()
    )


def test_required_local_runner_does_not_require_an_unselected_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("fugue.doctor._python_version_info", lambda: (3, 13, 0))
    monkeypatch.setattr("fugue.doctor.version", lambda name: "0.18.0")
    monkeypatch.setattr("fugue.doctor.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "fugue.doctor.resolve_console_script", lambda name: f"/venv/bin/{name}"
    )
    monkeypatch.setattr(
        "fugue.doctor._docker_status",
        lambda **_kwargs: {
            "cli_available": True,
            "daemon_available": True,
            "detail": "27.5.1",
            "network_ready": True,
            "network_detail": "probe passed",
        },
    )

    report = doctor_report(
        tmp_path,
        env={},
        required_capabilities=("local-runner",),
    )

    assert report["ok"] is True
    assert "model_credential" not in report["readiness"]["requirements"]


def test_required_local_runner_enforces_python_and_architecture(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("fugue.doctor._python_version_info", lambda: (3, 12, 9))
    monkeypatch.setattr("fugue.doctor._architecture", lambda: "riscv64")
    monkeypatch.setattr("fugue.doctor.version", lambda name: "0.18.0")
    monkeypatch.setattr("fugue.doctor.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "fugue.doctor.resolve_console_script", lambda name: f"/venv/bin/{name}"
    )
    monkeypatch.setattr(
        "fugue.doctor._docker_status",
        lambda **_kwargs: {
            "cli_available": True,
            "daemon_available": True,
            "detail": "27.5.1",
            "network_ready": True,
            "network_detail": "probe passed",
        },
    )

    report = doctor_report(
        tmp_path,
        model="anthropic/claude-sonnet-5",
        env={"ANTHROPIC_API_KEY": "present-for-test"},
        required_capabilities=("local-runner",),
    )

    assert report["ok"] is False
    requirements = report["readiness"]["requirements"]
    assert requirements["python_local_runner"]["ready"] is False
    assert requirements["host_architecture"]["ready"] is False


def test_doctor_rejects_unqualified_future_python(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("fugue.doctor._python_version_info", lambda: (3, 14, 0))

    report = doctor_report(tmp_path)

    assert report["python"]["supported"] is False
    assert report["python"]["local_runner_supported"] is False


def test_required_local_runner_rejects_unqualified_future_python(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("fugue.doctor._python_version_info", lambda: (3, 14, 0))
    monkeypatch.setattr("fugue.doctor.version", lambda name: "0.18.0")
    monkeypatch.setattr("fugue.doctor.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "fugue.doctor.resolve_console_script", lambda name: f"/venv/bin/{name}"
    )
    monkeypatch.setattr(
        "fugue.doctor._docker_status",
        lambda **_kwargs: {
            "cli_available": True,
            "daemon_available": True,
            "detail": "27.5.1",
            "network_ready": True,
            "network_detail": "probe passed",
        },
    )

    report = doctor_report(
        tmp_path,
        required_capabilities=("local-runner",),
    )

    assert report["ok"] is False
    assert report["readiness"]["requirements"]["python_local_runner"]["ready"] is False


def test_docker_network_probe_fails_closed_on_address_pool_exhaustion(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="all predefined address pools have been fully subnetted",
        )

    monkeypatch.setattr("fugue.doctor.subprocess.run", fake_run)

    from fugue.doctor import _probe_docker_network

    ready, detail = _probe_docker_network("/usr/bin/docker")

    assert ready is False
    assert "fully subnetted" in detail
    assert calls[0][:3] == ["/usr/bin/docker", "network", "create"]


def test_required_local_runner_cli_returns_nonzero_when_not_ready(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "fugue.doctor.doctor_report",
        lambda *args, **kwargs: {
            "ok": False,
            "readiness": {
                "mode": "required",
                "requested_capabilities": ["local-runner"],
                "ready": False,
                "requirements": {},
            },
            "distribution": {"version": "0.1", "digest": "f" * 64},
            "optional_features": {},
        },
    )

    exit_code = main(
        [
            "doctor",
            "--workspace",
            str(tmp_path),
            "--require",
            "local-runner",
            "--model",
            "anthropic/claude-sonnet-5",
            "--json",
        ]
    )

    assert exit_code == 2
    assert '"requested_capabilities": [' in capsys.readouterr().out


def test_required_local_runner_cli_reads_model_credential_from_env_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=credential-from-file\n")
    env_file.chmod(0o600)
    observed: dict[str, object] = {}

    def fake_report(*args, **kwargs):
        observed.update(kwargs)
        return {
            "ok": True,
            "readiness": {
                "mode": "required",
                "requested_capabilities": ["local-runner"],
                "ready": True,
                "requirements": {},
            },
            "distribution": {"version": "0.1", "digest": "f" * 64},
            "optional_features": {},
        }

    monkeypatch.setattr("fugue.doctor.doctor_report", fake_report)

    exit_code = main(
        [
            "doctor",
            "--workspace",
            str(tmp_path),
            "--require",
            "local-runner",
            "--model",
            "anthropic/claude-sonnet-5",
            "--env-file",
            str(env_file),
            "--json",
        ]
    )

    assert exit_code == 0
    assert observed["required_capabilities"] == ["local-runner"]
    assert observed["env"]["ANTHROPIC_API_KEY"] == "credential-from-file"
    capsys.readouterr()


def test_doctor_rejects_unknown_required_capability(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported required doctor capabilities"):
        doctor_report(tmp_path, required_capabilities=("unknown",))


def test_doctor_uses_supplied_environment_for_credentials(tmp_path: Path) -> None:
    report = doctor_report(
        tmp_path,
        model="anthropic/claude-sonnet-5",
        env={"ANTHROPIC_API_KEY": "present-for-test"},
    )

    assert report["model_route"]["selected"] is True
    assert report["model_route"]["credential_present"] is True
    assert report["credentials_present"]["ANTHROPIC_API_KEY"] is True
