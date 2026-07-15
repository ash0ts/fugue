from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fugue.bench.integrations import bind_integrations, load_integration
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
    assert binding.mcp_servers[0]["url"] == "http://retrieval:8000/mcp"
    compose_text = binding.compose_files[0].read_text()
    assert "secret-value" not in compose_text
    compose = yaml.safe_load(compose_text)
    service = compose["services"]["retrieval"]
    assert service["image"] == IMAGE
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert "ports" not in service


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
