from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from fugue.bench import services
from fugue.bench.services import (
    GRAPHITI_MANAGED_MARKER,
    GRAPHITI_SERVICE,
    managed_service_compose,
    managed_service_environment,
    managed_service_status,
    managed_services_for_systems,
    start_managed_services,
    stop_managed_services,
    without_managed_service_environment,
)


def _inspect_output(state: dict[str, object]) -> str:
    return "\n".join(
        (
            json.dumps(state),
            json.dumps(GRAPHITI_SERVICE.image),
            json.dumps({"io.fugue.managed-service": GRAPHITI_SERVICE.id}),
        )
    )


def test_graphiti_service_contract_is_pinned_and_secret_free() -> None:
    assert managed_services_for_systems(["none", "graphiti", "graphiti"]) == (
        GRAPHITI_SERVICE,
    )
    assert GRAPHITI_SERVICE.image == (
        "neo4j:5.26.12-community@"
        "sha256:9f75e8df4325a24f00fdd7a8c0bcce650a58375049b1058e496e8b43d6c36b37"
    )
    assert GRAPHITI_SERVICE.host_uri == "bolt://127.0.0.1:7687"
    assert GRAPHITI_SERVICE.container_uri == "bolt://host.docker.internal:7687"
    assert GRAPHITI_SERVICE.required_env_names == (
        "FUGUE_GRAPHITI_URI",
        "FUGUE_GRAPHITI_USER",
        "FUGUE_GRAPHITI_PASSWORD",
    )

    compose = managed_service_compose(GRAPHITI_SERVICE)
    service = compose["services"][GRAPHITI_SERVICE.id]
    assert service["container_name"] == "fugue-graphiti-neo4j"
    assert service["ports"] == ["127.0.0.1:7474:7474", "127.0.0.1:7687:7687"]
    assert service["volumes"] == ["fugue-graphiti-neo4j-data:/data"]
    assert service["healthcheck"]["test"][0] == "CMD-SHELL"
    assert service["environment"]["NEO4J_AUTH"] == (
        "${FUGUE_GRAPHITI_USER}/${FUGUE_GRAPHITI_PASSWORD}"
    )


def test_start_generates_private_credentials_and_waits_for_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        commands.append(command)
        if command[1] == "inspect":
            return subprocess.CompletedProcess(
                command,
                0,
                _inspect_output(
                    {
                        "Running": True,
                        "Status": "running",
                        "Health": {"Status": "healthy"},
                    }
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(services.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(services.subprocess, "run", run)

    [status] = start_managed_services(
        [GRAPHITI_SERVICE],
        repo_root=tmp_path,
        env={},
    )

    assert status.ready is True
    runtime = tmp_path / ".fugue/runtime/services" / GRAPHITI_SERVICE.id
    credentials_path = runtime / "credentials.json"
    credentials = json.loads(credentials_path.read_text())
    assert credentials["FUGUE_GRAPHITI_USER"] == "neo4j"
    assert len(credentials["FUGUE_GRAPHITI_PASSWORD"]) >= 32
    assert stat.S_IMODE(credentials_path.stat().st_mode) == 0o600
    compose_text = (runtime / "docker-compose.yaml").read_text()
    assert credentials["FUGUE_GRAPHITI_PASSWORD"] not in compose_text
    assert yaml.safe_load(compose_text) == managed_service_compose(GRAPHITI_SERVICE)
    up = next(command for command in commands if command[1:3] == ["compose", "-f"])
    assert up[-5:] == ["up", "-d", "--wait", "--wait-timeout", "120"]


def test_status_is_read_only_and_reports_docker_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(services.shutil, "which", lambda name: None)

    status = managed_service_status(GRAPHITI_SERVICE)

    assert status.state == "unavailable"
    assert status.ready is False
    assert not (tmp_path / ".fugue").exists()


def test_status_rejects_a_container_outside_the_pinned_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(services.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        services.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            "\n".join(
                (
                    json.dumps({"Running": True, "Health": {"Status": "healthy"}}),
                    json.dumps("neo4j:latest"),
                    json.dumps({}),
                )
            ),
            "",
        ),
    )

    status = managed_service_status(GRAPHITI_SERVICE)

    assert status.state == "unhealthy"
    assert status.ready is False
    assert "pinned managed-service contract" in status.detail


def test_managed_environment_uses_host_and_container_uris_only_when_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".fugue/runtime/services" / GRAPHITI_SERVICE.id
    runtime.mkdir(parents=True)
    credentials = {
        "FUGUE_GRAPHITI_USER": "neo4j",
        "FUGUE_GRAPHITI_PASSWORD": "private-password",
    }
    path = runtime / "credentials.json"
    path.write_text(json.dumps(credentials))
    path.chmod(0o600)
    monkeypatch.setattr(
        services,
        "managed_service_status",
        lambda spec: services.ManagedServiceStatus(
            spec.id,
            "healthy",
            True,
            "container is healthy",
            spec.container_name,
            spec.image,
            spec.host_uri,
        ),
    )

    host = managed_service_environment({}, repo_root=tmp_path)
    container = managed_service_environment(
        host,
        repo_root=tmp_path,
        target="container",
    )

    assert host["FUGUE_GRAPHITI_URI"] == GRAPHITI_SERVICE.host_uri
    assert container["FUGUE_GRAPHITI_URI"] == GRAPHITI_SERVICE.container_uri
    assert container["FUGUE_GRAPHITI_PASSWORD"] == "private-password"
    assert host[GRAPHITI_MANAGED_MARKER] == GRAPHITI_SERVICE.id
    assert without_managed_service_environment(host) == {}

    explicit = managed_service_environment(
        {"FUGUE_GRAPHITI_URI": "bolt+s://graph.example.test"},
        repo_root=tmp_path,
    )
    assert explicit == {"FUGUE_GRAPHITI_URI": "bolt+s://graph.example.test"}


def test_managed_environment_accepts_user_credentials_without_storing_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Environment resolution describes the selected service contract. Runtime
    # readiness is an independent preflight check and must not mutate a plan.
    monkeypatch.setattr(
        services,
        "managed_service_status",
        lambda spec: services.ManagedServiceStatus(
            spec.id,
            "stopped",
            False,
            "container is stopped",
            spec.container_name,
            spec.image,
            spec.host_uri,
        ),
    )

    resolved = managed_service_environment(
        {
            "FUGUE_GRAPHITI_USER": "neo4j",
            "FUGUE_GRAPHITI_PASSWORD": "user-supplied-password",
        },
        repo_root=tmp_path,
    )

    assert resolved["FUGUE_GRAPHITI_URI"] == GRAPHITI_SERVICE.host_uri
    assert resolved["FUGUE_GRAPHITI_PASSWORD"] == "user-supplied-password"
    assert resolved[GRAPHITI_MANAGED_MARKER] == GRAPHITI_SERVICE.id
    assert not (tmp_path / ".fugue").exists()


def test_stop_preserves_credentials_and_named_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".fugue/runtime/services" / GRAPHITI_SERVICE.id
    runtime.mkdir(parents=True)
    credentials_path = runtime / "credentials.json"
    credentials_path.write_text(
        json.dumps(
            {
                "FUGUE_GRAPHITI_USER": "neo4j",
                "FUGUE_GRAPHITI_PASSWORD": "private-password",
            }
        )
    )
    credentials_path.chmod(0o600)
    (runtime / "docker-compose.yaml").write_text(
        yaml.safe_dump(managed_service_compose(GRAPHITI_SERVICE))
    )
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        commands.append(command)
        if command[1] == "inspect":
            return subprocess.CompletedProcess(command, 1, "", "not found")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(services.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(services.subprocess, "run", run)

    [status] = stop_managed_services(
        [GRAPHITI_SERVICE],
        repo_root=tmp_path,
        env={},
    )

    down = commands[0]
    assert down[-4:] == ["down", "--remove-orphans", "--timeout", "30"]
    assert "-v" not in down and "--volumes" not in down
    assert credentials_path.is_file()
    assert status.state == "not_created"


def test_credential_file_rejects_broad_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".fugue/runtime/services" / GRAPHITI_SERVICE.id
    runtime.mkdir(parents=True)
    path = runtime / "credentials.json"
    path.write_text("{}\n")
    path.chmod(0o644)

    with pytest.raises(RuntimeError, match="mode 0600"):
        managed_service_environment({}, repo_root=tmp_path)


def test_service_runtime_rejects_symlink_traversal(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / ".fugue").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="remain inside the repository"):
        managed_service_environment({}, repo_root=tmp_path)
