from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fugue.bench.sandbox_policy import (
    attest_harbor_job,
    verify_harbor_job_attestation,
)


def _compose_service(tmp_path: Path) -> dict[str, object]:
    evidence = tmp_path / ".fugue/runtime/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    return {
        "image": "example/service@sha256:" + "a" * 64,
        "pull_policy": "never",
        "user": "65532:65532",
        "read_only": True,
        "network_mode": "service:main",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "deploy": {"resources": {"limits": {"cpus": "1.0", "memory": "1g"}}},
        "pids_limit": 512,
        "volumes": [
            {
                "type": "bind",
                "source": str(evidence),
                "target": "/evidence",
                "read_only": False,
            }
        ],
    }


def _config(
    tmp_path: Path,
    service: dict[str, object],
    *,
    service_name: str = "sidecar",
) -> dict[str, object]:
    compose = tmp_path / ".fugue/runtime/sidecar.yaml"
    compose.parent.mkdir(parents=True, exist_ok=True)
    compose.write_text(yaml.safe_dump({"services": {service_name: service}}))
    return {
        "environment": {"extra_docker_compose": [str(compose)]},
        "fugue": {},
    }


def _main_service() -> dict[str, object]:
    return {
        "pull_policy": "never",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "deploy": {"resources": {"limits": {"cpus": "8.0", "memory": "16g"}}},
        "pids_limit": 1024,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("privileged", True),
        ("pid", "host"),
        ("ipc", "host"),
        ("network_mode", "host"),
        ("devices", ["/dev/kvm"]),
        ("cap_add", ["SYS_ADMIN"]),
    ],
)
def test_harbor_policy_rejects_dangerous_compose_options(
    tmp_path: Path, field: str, value: object
) -> None:
    service = _compose_service(tmp_path)
    service[field] = value

    with pytest.raises(
        ValueError, match="forbidden options|main's network|host network"
    ):
        attest_harbor_job(
            _config(tmp_path, service),
            repo_root=tmp_path,
            bridge_required=False,
            require_files=True,
            strict_images=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("network_mode", "host", "host network"),
        ("security_opt", ["seccomp=unconfined"], "security profile"),
        ("cap_drop", [], "drop all capabilities"),
        ("deploy", {}, "CPU and memory limits"),
    ],
)
def test_harbor_policy_rejects_unsafe_main_service(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    main = _main_service()
    main[field] = value

    with pytest.raises(ValueError, match=message):
        attest_harbor_job(
            _config(tmp_path, main, service_name="main"),
            repo_root=tmp_path,
            bridge_required=False,
            require_files=True,
            strict_images=True,
        )


def test_harbor_policy_rejects_docker_socket_and_external_writable_bind(
    tmp_path: Path,
) -> None:
    service = _compose_service(tmp_path)
    service["volumes"] = ["/var/run/docker.sock:/var/run/docker.sock"]
    with pytest.raises(ValueError, match="Docker socket"):
        attest_harbor_job(
            _config(tmp_path, service),
            repo_root=tmp_path,
            bridge_required=False,
            require_files=True,
            strict_images=True,
        )

    external = tmp_path.parent / "application-source"
    external.mkdir(exist_ok=True)
    service["volumes"] = [f"{external}:/workspace"]
    with pytest.raises(ValueError, match="within the Fugue checkout"):
        attest_harbor_job(
            _config(tmp_path, service),
            repo_root=tmp_path,
            bridge_required=False,
            require_files=True,
            strict_images=True,
        )


def test_harbor_policy_rejects_unlocked_bridge_and_unpinned_image(
    tmp_path: Path,
) -> None:
    service = _compose_service(tmp_path)
    service["environment"] = {"UPSTREAM": "http://host.docker.internal:4000"}
    with pytest.raises(ValueError, match="locked bridge"):
        attest_harbor_job(
            _config(tmp_path, service),
            repo_root=tmp_path,
            bridge_required=False,
            require_files=True,
            strict_images=True,
        )

    service.pop("environment")
    service["image"] = "example/service:latest"
    with pytest.raises(ValueError, match="pinned by sha256"):
        attest_harbor_job(
            _config(tmp_path, service),
            repo_root=tmp_path,
            bridge_required=True,
            require_files=True,
            strict_images=True,
        )


def test_harbor_policy_rejects_unreviewed_compose_and_bind_options(
    tmp_path: Path,
) -> None:
    service = _compose_service(tmp_path)
    service["ports"] = ["8080:8080"]
    with pytest.raises(ValueError, match="unsupported options"):
        attest_harbor_job(
            _config(tmp_path, service),
            repo_root=tmp_path,
            bridge_required=False,
            require_files=True,
            strict_images=True,
        )

    service = _compose_service(tmp_path)
    mount = dict(service["volumes"][0])
    mount["bind"] = {"propagation": "rshared"}
    service["volumes"] = [mount]
    with pytest.raises(ValueError, match="only locked bind mounts"):
        attest_harbor_job(
            _config(tmp_path, service),
            repo_root=tmp_path,
            bridge_required=False,
            require_files=True,
            strict_images=True,
        )


def test_harbor_policy_rejects_changed_bridge_and_invalid_limits(
    tmp_path: Path,
) -> None:
    service = _compose_service(tmp_path)
    service["environment"] = {
        "FUGUE_BRIDGE_BASE_URL": "http://host.docker.internal:4999"
    }
    with pytest.raises(ValueError, match="locked endpoint exactly"):
        attest_harbor_job(
            _config(tmp_path, service),
            repo_root=tmp_path,
            bridge_required=True,
            require_files=True,
            strict_images=True,
        )

    service = _compose_service(tmp_path)
    service["deploy"] = {
        "resources": {"limits": {"cpus": "-1", "memory": "unbounded"}}
    }
    with pytest.raises(ValueError, match="invalid CPU limit"):
        attest_harbor_job(
            _config(tmp_path, service),
            repo_root=tmp_path,
            bridge_required=False,
            require_files=True,
            strict_images=True,
        )


def test_harbor_attestation_detects_compose_tampering_before_launch(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, _compose_service(tmp_path))
    attestation = attest_harbor_job(
        config,
        repo_root=tmp_path,
        bridge_required=False,
        require_files=True,
        strict_images=True,
    )
    config["fugue"]["sandbox_attestation"] = attestation.to_dict()
    config_path = tmp_path / ".fugue/runtime/job.json"
    config_path.write_text(json.dumps(config))
    verify_harbor_job_attestation(config_path, tmp_path)

    compose_path = Path(config["environment"]["extra_docker_compose"][0])
    raw = yaml.safe_load(compose_path.read_text())
    raw["services"]["sidecar"]["privileged"] = True
    compose_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="forbidden options"):
        verify_harbor_job_attestation(config_path, tmp_path)
