from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fugue.bench.integrations import (
    bind_integrations,
    effective_selections,
    load_integration,
)
from fugue.bench.library import IntegrationSelection

IMAGE = "ghcr.io/example/retrieval@sha256:" + "a" * 64


def test_compose_integration_is_pinned_hardened_and_secret_free(tmp_path: Path) -> None:
    path = tmp_path / "configs" / "fugue" / "integrations" / "retrieval.yaml"
    instructions = path.parent / "retrieval" / "usage.md"
    instructions.parent.mkdir(parents=True)
    instructions.write_text("# Retrieval\n")
    path.write_text(
        f"""
id: retrieval
version: "1"
support: experimental
runtime:
  type: compose
  image: {IMAGE}
  service: retrieval
  port: 8000
  healthcheck: {{path: /health}}
interfaces:
  - type: mcp
    name: retrieval
    transport: streamable-http
    path: /mcp
required_env: [RETRIEVAL_TOKEN]
instructions: [usage.md]
artifacts:
  - {{source: /logs/service.jsonl, service: retrieval}}
"""
    )

    binding = bind_integrations(
        [IntegrationSelection("retrieval")],
        repo_root=tmp_path,
        runtime_root=tmp_path / ".fugue" / "runtime" / "unit",
        job_name="job",
        env={"RETRIEVAL_TOKEN": "secret-value"},
        write=True,
    )

    assert binding.applicable
    assert binding.env == {"RETRIEVAL_TOKEN": "${RETRIEVAL_TOKEN}"}
    assert binding.mcp_servers[0]["url"] == "http://127.0.0.1:8000/mcp"
    compose_text = binding.compose_files[0].read_text()
    assert "secret-value" not in compose_text
    compose = yaml.safe_load(compose_text)
    service = compose["services"]["retrieval"]
    assert service["image"] == IMAGE
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["user"] == "65532:65532"
    assert service["network_mode"] == "service:main"
    assert service["tmpfs"] == ["/tmp:rw,noexec,nosuid,size=64m"]
    assert service["healthcheck"]["test"][:3] == ["CMD", "python", "-c"]
    assert "ports" not in service
    assert "main" not in compose["services"]


def test_external_integration_requires_host_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir(parents=True)
    (root / "external.yaml").write_text(
        """
id: external
version: "1"
runtime: {type: external, url: https://mcp.example.test}
interfaces:
  - {type: mcp, name: external, transport: streamable-http, path: /mcp}
"""
    )
    with pytest.raises(ValueError, match="allowed_hosts"):
        load_integration("external", tmp_path)


def test_http_tool_allowlist_requires_a_policy_gateway(tmp_path: Path) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir(parents=True)
    (root / "external.yaml").write_text(
        """
id: external
version: "1"
runtime: {type: external, url: https://mcp.example.test}
allowed_hosts: [mcp.example.test]
interfaces:
  - type: mcp
    name: external
    transport: streamable-http
    path: /mcp
    allowed_tools: [search]
"""
    )
    with pytest.raises(ValueError, match="policy gateway"):
        load_integration("external", tmp_path)


def test_http_interface_exposes_only_a_declared_endpoint(tmp_path: Path) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir(parents=True)
    (root / "api.yaml").write_text(
        """
id: api
version: "1"
runtime: {type: external, url: https://api.example.test}
allowed_hosts: [api.example.test]
interfaces:
  - {type: http, name: search, path: /v1/search}
"""
    )

    binding = bind_integrations(
        [IntegrationSelection("api")],
        repo_root=tmp_path,
        runtime_root=tmp_path / ".fugue" / "runtime",
        job_name="job",
        env={},
        write=False,
    )

    assert binding.env == {
        "FUGUE_INTEGRATION_API_SEARCH_URL": "https://api.example.test/v1/search"
    }
    assert binding.allowed_hosts == ("api.example.test",)


def test_selection_config_rejects_secret_fields(tmp_path: Path) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir(parents=True)
    (root / "api.yaml").write_text(
        """
id: api
version: "1"
runtime: {type: external, url: https://api.example.test}
allowed_hosts: [api.example.test]
interfaces:
  - {type: http, name: search, path: /v1/search}
config_schema:
  api_key: {type: string}
"""
    )
    with pytest.raises(ValueError, match="required_env"):
        bind_integrations(
            [IntegrationSelection("api", {"api_key": "secret"})],
            repo_root=tmp_path,
            runtime_root=tmp_path / ".fugue" / "runtime",
            job_name="job",
            env={},
            write=False,
        )


def test_integration_instructions_reject_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    instructions = root / "external"
    instructions.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("unreviewed\n")
    (instructions / "usage.md").symlink_to(outside)
    (root / "external.yaml").write_text(
        """
id: external
version: "1"
runtime: {type: external, url: https://api.example.test}
allowed_hosts: [api.example.test]
interfaces:
  - {type: http, name: endpoint, path: /}
instructions: [usage.md]
"""
    )

    with pytest.raises(ValueError, match="may not use symlinks"):
        bind_integrations(
            [IntegrationSelection("external")],
            repo_root=tmp_path,
            runtime_root=tmp_path / "runtime",
            job_name="job",
            env={},
            write=False,
        )


def test_missing_environment_and_disabled_integrations_are_not_applicable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir(parents=True)
    (root / "required.yaml").write_text(
        f"""
id: required
version: "1"
runtime: {{type: compose, image: {IMAGE}, service: required, port: 8000}}
interfaces:
  - {{type: http, name: endpoint, path: /}}
required_env: [SERVICE_TOKEN]
"""
    )
    (root / "disabled.yaml").write_text(
        """
id: disabled
version: "1"
support: disabled
runtime: {type: external, url: https://api.example.test}
allowed_hosts: [api.example.test]
interfaces:
  - {type: http, name: endpoint, path: /}
"""
    )

    missing = bind_integrations(
        [IntegrationSelection("required")],
        repo_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        job_name="job",
        env={},
        write=True,
    )
    disabled = bind_integrations(
        [IntegrationSelection("disabled")],
        repo_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        job_name="job",
        env={},
        write=True,
    )

    assert not missing.applicable
    assert missing.skip_reason == "integration required requires environment: SERVICE_TOKEN"
    assert not missing.compose_files
    assert not disabled.applicable
    assert disabled.skip_reason == "integration disabled is disabled"


def test_builtin_stdio_allowlist_and_variant_override_are_explicit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir(parents=True)
    (root / "stdio.yaml").write_text(
        """
id: stdio
version: "1"
runtime:
  type: builtin
  command: [python, -m, example_server]
interfaces:
  - type: mcp
    name: reviewed
    transport: stdio
    allowed_tools: [search, read]
"""
    )

    binding = bind_integrations(
        [IntegrationSelection("stdio")],
        repo_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        job_name="job",
        env={},
        write=False,
    )

    assert binding.mcp_servers == (
        {
            "name": "reviewed",
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "example_server"],
            "fugue_allowed_tools": ["search", "read"],
        },
    )
    inherited = [IntegrationSelection("stdio")]
    assert effective_selections(inherited, None) == inherited
    assert effective_selections(inherited, []) == []


@pytest.mark.parametrize(
    ("runtime", "interface", "message"),
    [
        (
            f"{{type: compose, image: {IMAGE}, port: 8000, url: https://api.example.test}}",
            "{type: http, name: endpoint, path: /}",
            "may not declare an external URL",
        ),
        (
            "{type: external, url: https://api.example.test, command: [server]}",
            "{type: http, name: endpoint, path: /}",
            "accepts only type and url",
        ),
        (
            "{type: builtin, command: [server]}",
            "{type: http, name: endpoint, url: https://api.example.test}",
            "only stdio MCP",
        ),
        (
            f"{{type: compose, image: {IMAGE}, port: 8000}}",
            "{type: mcp, name: endpoint, transport: stdio}",
            "may not use stdio",
        ),
    ],
)
def test_runtime_and_interface_combinations_are_strict(
    tmp_path: Path, runtime: str, interface: str, message: str
) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir(parents=True)
    (root / "invalid.yaml").write_text(
        f"""
id: invalid
version: "1"
runtime: {runtime}
allowed_hosts: [api.example.test]
interfaces:
  - {interface}
"""
    )

    with pytest.raises(ValueError, match=message):
        load_integration("invalid", tmp_path)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1",
        "https://10.0.0.1",
        "https://169.254.169.254",
        "https://192.168.1.1",
    ],
)
def test_external_integrations_reject_non_public_ip_literals(
    tmp_path: Path, url: str
) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir(parents=True)
    host = url.removeprefix("https://")
    (root / "external.yaml").write_text(
        f"""
id: external
version: "1"
runtime: {{type: external, url: {url}}}
allowed_hosts: [{host}]
interfaces:
  - {{type: http, name: endpoint, path: /}}
"""
    )

    with pytest.raises(ValueError, match="non-public IP"):
        load_integration("external", tmp_path)


def test_config_schema_enforces_required_types_and_artifact_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir(parents=True)
    (root / "typed.yaml").write_text(
        f"""
id: typed
version: "1"
runtime: {{type: compose, image: {IMAGE}, service: typed, port: 8000}}
interfaces:
  - {{type: http, name: endpoint, path: /}}
artifacts:
  - {{source: /logs/events.jsonl, service: typed}}
config_schema:
  top_k: {{type: integer, required: true}}
  explain: {{type: boolean}}
"""
    )

    with pytest.raises(ValueError, match="requires config field top_k"):
        bind_integrations(
            [IntegrationSelection("typed")],
            repo_root=tmp_path,
            runtime_root=tmp_path / "runtime",
            job_name="job",
            env={},
            write=False,
        )
    with pytest.raises(ValueError, match="top_k must be integer"):
        bind_integrations(
            [IntegrationSelection("typed", {"top_k": True})],
            repo_root=tmp_path,
            runtime_root=tmp_path / "runtime",
            job_name="job",
            env={},
            write=False,
        )

    binding = bind_integrations(
        [IntegrationSelection("typed", {"top_k": 5, "explain": False})],
        repo_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        job_name="job",
        env={},
        write=False,
    )
    assert binding.artifacts == ({"source": "/logs/events.jsonl", "service": "typed"},)


def test_compose_integrations_reject_shared_namespace_port_collisions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "configs" / "fugue" / "integrations"
    root.mkdir(parents=True)
    for integration_id in ("one", "two"):
        (root / f"{integration_id}.yaml").write_text(
            f"""
id: {integration_id}
version: "1"
runtime: {{type: compose, image: {IMAGE}, service: {integration_id}, port: 8000}}
interfaces:
  - {{type: http, name: {integration_id}, path: /}}
"""
        )

    with pytest.raises(ValueError, match="both use port 8000"):
        bind_integrations(
            [IntegrationSelection("one"), IntegrationSelection("two")],
            repo_root=tmp_path,
            runtime_root=tmp_path / "runtime",
            job_name="job",
            env={},
            write=False,
        )
