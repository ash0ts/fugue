from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from fugue.bench.candidates import resolve_candidate
from fugue.bench.coreweave import (
    COREWEAVE_ENVIRONMENT_IMPORT,
    CoreWeaveGatewayLockV1,
    CoreWeaveRuntimeManifestV1,
    bind_coreweave_job,
    build_coreweave_lock,
    classify_coreweave_failure,
    comparison_environment,
    coreweave_cell_environment,
    coreweave_execution_identity,
    coreweave_gateway_mcp_servers,
    coreweave_lock_from_dict,
    require_coreweave_runtime_assets,
    runtime_manifest_from_dict,
    validate_coreweave_artifact_archive,
    validate_effective_attestation,
    write_coreweave_lock,
)
from fugue.connectivity_gateway import GatewayPolicyV1

IMAGE = "ghcr.io/example/fugue@sha256:" + "a" * 64
PROFILE_ID = "profile-123"
RUNNER_ID = "runner-123"


def _runtime_manifest() -> CoreWeaveRuntimeManifestV1:
    raw = {
        "schema_version": 1,
        "python_version": "3.13",
        "assets": [
            {
                "kind": "fugue",
                "id": "source",
                "digest": "d" * 64,
                "path": "/fugue-src/fugue",
            }
        ],
        "manifest_sha256": "",
    }
    from fugue.bench.coreweave import runtime_manifest_from_dict

    return runtime_manifest_from_dict(raw)


def _profile(*, network: str = "none") -> bytes:
    value = yaml.safe_load(
        Path("deploy/coreweave/fugue-untrusted-ci-v1.yaml").read_text()
    )
    value["spec"]["container_image"] = IMAGE
    if network == "gateway":
        value["spec"]["network"]["egress"]["modes"]["fugue-gateway"]["cidrs"] = [
            "203.0.113.7/32"
        ]
    return yaml.safe_dump(value, sort_keys=False).encode()


def _lock(*, network: str = "none"):
    gateway = (
        CoreWeaveGatewayLockV1(
            base_url="https://gateway.example.com",
            policy_sha256="b" * 64,
            route_ids=("model", "wandb-api", "weave", "mcp-wandb"),
            certificate_sha256="c" * 64,
        )
        if network == "gateway"
        else None
    )
    return build_coreweave_lock(
        runner_id=RUNNER_ID,
        profile_id=PROFILE_ID,
        profile_name="fugue-untrusted-ci-v1",
        profile_document=_profile(network=network),
        image=IMAGE,
        runtime_manifest=_runtime_manifest(),
        network=network,  # type: ignore[arg-type]
        gateway=gateway,
        gateway_cidrs=("203.0.113.7/32",) if network == "gateway" else (),
        gateway_hosts=("gateway.example.com",) if network == "gateway" else (),
        max_lifetime_seconds=1800,
    )


def test_coreweave_lock_is_strict_and_content_addressed(tmp_path: Path) -> None:
    lock = _lock()
    path = tmp_path / "lock.json"
    written = write_coreweave_lock(path, lock)

    assert written.lock_sha256
    assert path.stat().st_mode & 0o777 == 0o600
    raw = written.to_dict()
    raw["runtime_class"] = "runc"
    with pytest.raises(ValueError, match="kata-qemu"):
        coreweave_lock_from_dict(raw)

    raw = written.to_dict()
    raw["surprise"] = True
    with pytest.raises(ValueError, match="unknown CoreWeave sandbox lock"):
        coreweave_lock_from_dict(raw)


def test_runtime_manifest_is_strict_and_required_assets_must_match() -> None:
    manifest = _runtime_manifest()
    assert manifest.manifest_sha256
    lock = _lock()
    require_coreweave_runtime_assets(
        lock,
        [{"kind": "fugue", "id": "source", "digest": "sha256:" + "d" * 64}],
    )
    with pytest.raises(ValueError, match="missing locked assets"):
        require_coreweave_runtime_assets(
            lock,
            [{"kind": "skill", "id": "missing", "digest": "e" * 64}],
        )
    with pytest.raises(ValueError, match="digests differ"):
        require_coreweave_runtime_assets(
            lock,
            [{"kind": "fugue", "id": "source", "digest": "e" * 64}],
        )
    raw = manifest.to_dict()
    raw["assets"][0]["path"] = "/workspace/mutable"
    raw["manifest_sha256"] = ""
    with pytest.raises(ValueError, match="locked image asset root"):
        runtime_manifest_from_dict(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["spec"].update(runtime_class="runc"), "kata-qemu"),
        (
            lambda value: value["spec"]["namespace"].update(strategy="per-org"),
            "per-user",
        ),
        (
            lambda value: value["spec"]["network"].update(
                ingress={"public": {"scope": "any"}}
            ),
            "ingress",
        ),
        (
            lambda value: value["spec"]["pod"]["spec"].update(hostNetwork=True),
            "host namespaces",
        ),
        (
            lambda value: value["spec"]["pod"]["spec"]["containers"][0][
                "securityContext"
            ].update(allowPrivilegeEscalation=True),
            "security controls",
        ),
    ],
)
def test_coreweave_profile_rejects_security_downgrades(
    mutation, message: str
) -> None:
    profile = yaml.safe_load(_profile())
    mutation(profile)
    with pytest.raises(ValueError, match=message):
        build_coreweave_lock(
            runner_id=RUNNER_ID,
            profile_id=PROFILE_ID,
            profile_name="fugue-untrusted-ci-v1",
            profile_document=yaml.safe_dump(profile).encode(),
            image=IMAGE,
            runtime_manifest=_runtime_manifest(),
            network="none",
            gateway=None,
        )


def test_coreweave_is_execution_identity_not_candidate_material() -> None:
    environment = comparison_environment(
        _lock(network="gateway"),
        network="gateway",
        max_lifetime_seconds=1800,
    )
    identity = coreweave_execution_identity(environment)

    assert environment["import_path"] == COREWEAVE_ENVIRONMENT_IMPORT
    assert identity is not None
    assert identity["backend"] == "coreweave"
    assert identity["profile_id"] == PROFILE_ID
    assert identity["network"]["selected_mode"] == "gateway"
    assert identity["lock"]["lock_sha256"] == identity["lock_sha256"]
    local = resolve_candidate(
        harness="codex",
        harness_version="1",
        model_route={"provider": "wandb", "model": "model"},
        prompt_digest=None,
        skills=[],
        context={"id": "none"},
        integrations=[],
        agent={},
        execution={"backend": "local"},
    )
    remote = resolve_candidate(
        harness="codex",
        harness_version="1",
        model_route={"provider": "wandb", "model": "model"},
        prompt_digest=None,
        skills=[],
        context={"id": "none"},
        integrations=[],
        agent={},
        execution={"sandbox_runtime": identity},
    )
    assert local.candidate_id == remote.candidate_id
    assert local.execution_fingerprint != remote.execution_fingerprint


def test_coreweave_attestation_fails_on_effective_drift() -> None:
    lock = _lock(network="gateway")
    attestation = validate_effective_attestation(
        lock,
        sandbox_id="sandbox-1",
        runner_id="other-runner",
        profile_id=PROFILE_ID,
        applied_egress_mode="internet",
        applied_ingress_mode="public",
        resource_requests={"cpu": "1", "memory": "1Gi"},
        resource_limits={"cpu": "8", "memory": "32Gi"},
    )

    assert attestation.eligible is False
    assert set(attestation.failures) == {
        "runner differs from lock",
        "applied egress differs from lock",
        "sandbox unexpectedly exposes ingress",
        "resource requests differ from lock",
        "resource limits differ from lock",
    }


def test_cell_gateway_token_is_minted_only_at_execution_admission(
    tmp_path: Path,
) -> None:
    lock = _lock(network="gateway")
    environment = comparison_environment(
        lock, network="gateway", max_lifetime_seconds=1800
    )
    environment = bind_coreweave_job(
        environment,
        instance_id="instance-1",
        run_id="run-1",
        job_name="job-1",
        execution_fingerprint="d" * 64,
    )
    config = tmp_path / "job.json"
    config.write_text(json.dumps({"environment": environment}))
    signing_key = b"k" * 32

    overlay = coreweave_cell_environment(
        config_path=config,
        cell_id="cell-1",
        run_id="run-1",
        execution_fingerprint="d" * 64,
        worker_env={
            "FUGUE_INSTANCE_ID": "instance-1",
            "FUGUE_GATEWAY_SIGNING_KEY": base64.b64encode(signing_key).decode(),
        },
    )

    assert overlay["FUGUE_CELL_ID"] == "cell-1"
    assert overlay["FUGUE_CONNECTIVITY_TOKEN"]
    assert "FUGUE_GATEWAY_SIGNING_KEY" not in overlay
    assert "WANDB_API_KEY" not in overlay


def test_remote_mcp_is_rewritten_to_its_locked_gateway_route() -> None:
    environment = comparison_environment(
        _lock(network="gateway"),
        network="gateway",
        max_lifetime_seconds=1800,
    )
    servers = coreweave_gateway_mcp_servers(
        environment,
        [
            {
                "name": "wandb",
                "transport": "streamable-http",
                "url": "https://mcp.example.com/mcp",
                "integration_id": "wandb",
            },
            {
                "name": "local",
                "transport": "stdio",
                "command": "/fugue-components/local/bin/server",
                "integration_id": "local",
            },
        ],
    )

    assert servers[0]["url"] == (
        "https://gateway.example.com/routes/mcp-wandb/mcp"
    )
    assert servers[1]["command"] == "/fugue-components/local/bin/server"

    raw = _lock(network="gateway").to_dict()
    assert raw["gateway"] is not None
    raw["gateway"]["route_ids"] = tuple(
        route_id
        for route_id in raw["gateway"]["route_ids"]
        if route_id != "mcp-wandb"
    )
    raw["gateway"]["policy_sha256"] = "e" * 64
    raw["lock_sha256"] = ""
    drifted = coreweave_lock_from_dict(raw)
    changed = comparison_environment(
        drifted,
        network="gateway",
        max_lifetime_seconds=1800,
    )
    with pytest.raises(ValueError, match="missing remote MCP route"):
        coreweave_gateway_mcp_servers(changed, [servers[0]])


def test_gateway_lock_type_is_not_accidentally_a_policy() -> None:
    # A small guard against serializing the operator's full policy into a cell.
    assert "routes" not in CoreWeaveGatewayLockV1(
        base_url="https://gateway.example.com",
        policy_sha256="b" * 64,
        route_ids=("model",),
        certificate_sha256="c" * 64,
    ).to_dict()
    assert GatewayPolicyV1.__name__ == "GatewayPolicyV1"


def _write_archive(
    path: Path,
    *,
    name: str,
    entry_type: bytes = tarfile.REGTYPE,
    data: bytes = b"ok",
) -> None:
    info = tarfile.TarInfo(name)
    info.type = entry_type
    if entry_type == tarfile.REGTYPE:
        info.size = len(data)
    elif entry_type == tarfile.SYMTYPE:
        info.linkname = "/etc/passwd"
        data = b""
    else:
        data = b""
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(data) if data else None)


def test_coreweave_artifact_archive_accepts_only_bounded_regular_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "artifacts.tar.gz"
    _write_archive(archive, name="./result.json")
    members = validate_coreweave_artifact_archive(archive)
    assert [member.name for member in members] == ["./result.json"]

    _write_archive(archive, name="../../etc/passwd")
    with pytest.raises(RuntimeError, match="path traversal"):
        validate_coreweave_artifact_archive(archive)

    _write_archive(
        archive,
        name="./link",
        entry_type=tarfile.SYMTYPE,
    )
    with pytest.raises(RuntimeError, match="non-regular"):
        validate_coreweave_artifact_archive(archive)

    _write_archive(archive, name="./large", data=b"abc")
    monkeypatch.setattr("fugue.bench.coreweave._MAX_ARTIFACT_BYTES", 2)
    with pytest.raises(RuntimeError, match="expanded byte limit"):
        validate_coreweave_artifact_archive(archive)


def test_coreweave_artifact_archive_rejects_malformed_input(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "artifacts.tar.gz"
    archive.write_bytes(b"not a tar archive")
    with pytest.raises(RuntimeError, match="malformed"):
        validate_coreweave_artifact_archive(archive)


def test_coreweave_failure_is_normalized_and_redacted() -> None:
    failure = classify_coreweave_failure(
        RuntimeError(
            "gateway token=do-not-publish Bearer also-do-not-publish"
        ),
        "gateway",
    )

    assert failure.category == "gateway"
    assert "do-not-publish" not in failure.detail
    assert failure.retryable_before_agent_start is False
