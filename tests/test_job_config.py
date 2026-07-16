from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fugue.bench import services
from fugue.bench.job_config import render_jobs
from fugue.bench.library import (
    ContextSelection,
    ExperimentSpec,
    FeatureVariant,
    IntegrationSelection,
    save_prompt,
    save_skill,
)
from fugue.bench.manifest import load_manifest
from fugue.bench.services import GRAPHITI_SERVICE, ManagedServiceStatus


def test_latin_square_renders_one_harness_per_variant_task_coordinate(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "matrix.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: hermes, agent: fugue.agents:FugueHermes}
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
  - {id: task-b}
"""
    )
    experiment = ExperimentSpec(
        id="matrix",
        title="Matrix",
        variants=[
            FeatureVariant(
                id="none",
                label="None",
                context=ContextSelection(system_id="none"),
            ),
            FeatureVariant(
                id="agents",
                label="Agents",
                context=ContextSelection(system_id="agentsmd"),
            ),
        ],
    )

    jobs = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="latin",
        harness_assignment="latin_square",
    )

    assert {(job.variant_id, job.task_id, job.harness) for job in jobs} == {
        ("none", "task-a", "hermes"),
        ("none", "task-b", "codex"),
        ("agents", "task-a", "codex"),
        ("agents", "task-b", "hermes"),
    }
    selected = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="selected",
        variant_names=["agents"],
        harness_assignment="latin_square",
    )
    assert len(selected) == 2
    assert {job.variant_id for job in selected} == {"agents"}


def test_graphiti_job_uses_container_uri_without_serializing_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    context_path = tmp_path / "configs/fugue/context-systems/graphiti.yaml"
    context_path.parent.mkdir(parents=True)
    context_path.write_text(
        """
id: graphiti
title: Graphiti
provider: fugue.bench.context:EmptyContextProvider
version: test
capabilities: [prepare, retrieve, bind]
deliveries: [portable, native_mcp]
required_env: [FUGUE_GRAPHITI_URI, FUGUE_GRAPHITI_USER, FUGUE_GRAPHITI_PASSWORD]
config:
  binding:
    managed_runtime: fugue_context
    mcp_servers:
      - {name: fugue-memory, command: python, args: [-m, fugue.context_server]}
"""
    )
    credentials_dir = tmp_path / ".fugue/runtime/services" / GRAPHITI_SERVICE.id
    credentials_dir.mkdir(parents=True)
    credentials_path = credentials_dir / "credentials.json"
    credentials_path.write_text(
        json.dumps(
            {
                "FUGUE_GRAPHITI_USER": "neo4j",
                "FUGUE_GRAPHITI_PASSWORD": "private-password",
            }
        )
    )
    credentials_path.chmod(0o600)
    monkeypatch.setattr(
        services,
        "managed_service_status",
        lambda spec: ManagedServiceStatus(
            spec.id,
            "healthy",
            True,
            "container is healthy",
            spec.container_name,
            spec.image,
            spec.host_uri,
        ),
    )
    env = services.managed_service_environment({}, repo_root=tmp_path)
    experiment = ExperimentSpec(
        id="graphiti-a",
        title="Graphiti A",
        variants=[
            FeatureVariant(
                id="graphiti",
                label="Graphiti",
                context=ContextSelection(system_id="graphiti"),
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env=env,
        model="openai/gpt-5",
        run_id="unit",
    )

    agent_env = job.config["agents"][0]["env"]
    assert agent_env["FUGUE_GRAPHITI_URI"] == "${FUGUE_GRAPHITI_URI}"
    assert agent_env["FUGUE_GRAPHITI_PASSWORD"] == "${FUGUE_GRAPHITI_PASSWORD}"
    assert job.env["FUGUE_GRAPHITI_URI"] == GRAPHITI_SERVICE.container_uri
    assert job.env["FUGUE_GRAPHITI_PASSWORD"] == "private-password"
    assert "private-password" not in job.config_path.read_text()


def test_render_jobs_writes_harbor_config_without_secrets(tmp_path: Path):
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset:
  ref: swe-bench/swe-bench-verified
harnesses:
  - name: codex
    agent: fugue.agents:FugueCodex
tasks:
  - id: astropy__astropy-12907
    expected_paths: [astropy/modeling/separable.py]
"""
    )
    save_prompt("prompt-a", "# Prompt A\n", tmp_path)
    save_skill("skill-a", "# Skill A\n", tmp_path)
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        manifest=manifest_path,
        variants=[
            FeatureVariant(id="baseline", label="Baseline"),
            FeatureVariant(
                id="prompt-skill",
                label="Prompt + skill",
                prompt_id="prompt-a",
                skills=["skill-a"],
                context=ContextSelection(system_id="agentsmd"),
                agent_kwargs={"temperature": 0},
                agent_env={"FUGUE_AGENT_MODE": "strict"},
                environment={"override_memory_mb": 4096},
                verifier={"type": "pytest"},
            ),
        ],
        environment={
            "type": "docker",
            "cpu_enforcement_policy": "limit",
            "memory_enforcement_policy": "guarantee",
            "override_memory_mb": 2048,
        },
        artifacts=["/logs/artifacts"],
    )
    env = {
        "OPENAI_API_KEY": "secret-openai",
        "WANDB_API_KEY": "secret-wandb",
    }

    rendered = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env=env,
        model="openai/gpt-5",
        run_id="unit",
    )

    assert len(rendered) == 2
    job = next(item for item in rendered if item.variant_id == "prompt-skill")
    assert job.command == ["harbor", "run", "--config", job.config_path.as_posix()]
    assert job.prompt_id == "prompt-a"
    assert job.context_system_id == "agentsmd"
    assert job.context_version == "1"
    assert set(job.context_cache_keys) == {"astropy__astropy-12907"}
    assert job.context_cache_ready is False
    assert len(job.agent_config_hash) == 64
    config_text = job.config_path.read_text()
    assert "secret-openai" not in config_text
    assert "secret-wandb" not in config_text
    config = json.loads(config_text)
    assert config["agents"][0]["import_path"] == "fugue.agents:FugueCodex"
    assert "name" not in config["agents"][0]
    assert config["agents"][0]["model_name"] == "openai/gpt-5"
    assert config["agents"][0]["skills"] == ["configs/fugue/skills/skill-a"]
    assert config["agents"][0]["kwargs"] == {"temperature": 0}
    assert config["agents"][0]["env"] == {"FUGUE_AGENT_MODE": "strict"}
    instruction_paths = config["extra_instruction_paths"]
    assert (
        instruction_paths[0] == ".fugue/runtime/unit/context-instructions/agentsmd.md"
    )
    assert instruction_paths[-1] == "configs/fugue/prompts/prompt-a.md"
    assert any(path.endswith("/artifact/AGENTS.md") for path in instruction_paths)
    assert config["environment"]["override_memory_mb"] == 4096
    assert config["environment"]["type"] == "docker"
    assert config["artifacts"] == ["/logs/artifacts"]
    assert any(
        mount["target"] == "/fugue-context" for mount in config["environment"]["mounts"]
    )
    assert config["verifier"] == {"type": "pytest"}
    assert config["fugue"]["experiment_id"] == "experiment-a"
    assert config["fugue"]["variant_id"] == "prompt-skill"
    assert config["fugue"]["prompt_id"] == "prompt-a"
    assert config["fugue"]["context_system_id"] == "agentsmd"
    assert config["fugue"]["context_runtime_required"] is False
    assert config["fugue"]["context_version"] == "1"
    assert len(config["fugue"]["context_config_hash"]) == 64
    assert config["fugue"]["agent_config_hash"] == job.agent_config_hash
    assert config["n_attempts"] == 1
    assert config["fugue"]["trial_index"] == 1
    assert config["fugue"]["comparison_example_id"] == job.comparison_example_id
    assert "expected_evidence_paths" not in config["fugue"]
    assert "FUGUE_EXPECTED_EVIDENCE_PATHS" not in job.env
    assert job.expected_evidence_paths == ("astropy/modeling/separable.py",)
    assert config["fugue"]["candidate_id"] == job.candidate_id
    assert job.env["FUGUE_IDENTITY_SCHEMA_VERSION"] == "1"
    assert job.resolved_candidate.definition["harness_version"] == (
        "codex@0.143.0+fugue-flat-mcp.1+weave-codex@0.1.1+fugue-mcp-meta.1+skill-use.1"
    )
    assert job.resolved_candidate.definition["model_route"][
        "tool_result_modalities"
    ] == ["text", "image"]
    assert job.resolved_candidate.definition["harness"] == "codex"
    assert config["fugue"]["trace_content"] == "full"
    assert job.env["FUGUE_TRACE_CONTENT"] == "full"


def test_task_outputs_do_not_change_candidate_identity_or_overlap_harbor_artifacts(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "tasks.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses: [{name: codex, agent: fugue.agents:FugueCodex}]
tasks:
  - {id: first, artifacts: [/root/first.json]}
  - {id: second, artifacts: [/root/second.json]}
"""
    )
    experiment = ExperimentSpec(
        id="task-artifacts",
        title="Task artifacts",
        variants=[FeatureVariant(id="baseline", label="Baseline")],
    )

    jobs = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
        workload_artifacts=["/logs/artifacts/fugue-answer.md"],
    )

    assert len(jobs) == 2
    assert jobs[0].candidate_id == jobs[1].candidate_id
    assert jobs[0].agent_config_hash == jobs[1].agent_config_hash
    [with_collection] = render_jobs(
        experiment=ExperimentSpec(
            id="task-artifacts-renamed",
            title="Task artifacts renamed",
            variants=[FeatureVariant(id="renamed", label="Renamed")],
            artifacts=["/root/operator-only.json"],
        ),
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit-renamed",
        n_tasks=1,
    )
    assert with_collection.candidate_id == jobs[0].candidate_id
    assert with_collection.agent_config_hash == jobs[0].agent_config_hash
    for job, output in zip(
        jobs, ("/root/first.json", "/root/second.json"), strict=True
    ):
        assert job.config["artifacts"] == [output, "/logs/artifacts"]
        assert job.config["fugue"]["expected_artifact_paths"] == [
            output,
            "/logs/artifacts/fugue-answer.md",
        ]
        assert json.loads(job.env["FUGUE_EXPECTED_ARTIFACT_PATHS"]) == [
            output,
            "/logs/artifacts/fugue-answer.md",
        ]


def test_render_job_binds_explicit_integration_without_serializing_secret(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    integration = tmp_path / "configs" / "fugue" / "integrations" / "retrieval.yaml"
    integration.parent.mkdir(parents=True)
    integration.write_text(
        """
id: retrieval
version: "1"
support: experimental
runtime:
  type: compose
  image: ghcr.io/example/retrieval@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  service: retrieval
  port: 8000
interfaces:
  - {type: mcp, name: retrieval, transport: streamable-http, path: /mcp}
required_env: [RETRIEVAL_TOKEN]
"""
    )
    experiment = ExperimentSpec(
        id="integration-a",
        title="Integration A",
        variants=[
            FeatureVariant(
                id="retrieval",
                label="Retrieval",
                integrations=[IntegrationSelection("retrieval")],
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={"RETRIEVAL_TOKEN": "secret-value"},
        model="openai/gpt-5",
        run_id="unit",
    )

    agent = job.config["agents"][0]
    assert agent["env"] == {"RETRIEVAL_TOKEN": "${RETRIEVAL_TOKEN}"}
    assert agent["mcp_servers"] == [
        {
            "name": "retrieval",
            "transport": "streamable-http",
            "url": "http://127.0.0.1:8000/mcp",
        }
    ]
    assert "secret-value" not in job.config_path.read_text()
    assert job.env["RETRIEVAL_TOKEN"] == "secret-value"
    assert job.integration_ids == ("retrieval",)
    assert job.config["fugue"]["integrations"][0]["support"] == "experimental"


def test_missing_integration_secret_marks_cell_not_applicable_without_injection(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    integration = tmp_path / "configs" / "fugue" / "integrations" / "retrieval.yaml"
    integration.parent.mkdir(parents=True)
    integration.write_text(
        """
id: retrieval
version: "1"
runtime: {type: external, url: https://api.example.test}
allowed_hosts: [api.example.test]
interfaces:
  - {type: http, name: search, path: /search}
required_env: [RETRIEVAL_TOKEN]
"""
    )
    experiment = ExperimentSpec(
        id="missing-secret",
        title="Missing secret",
        variants=[
            FeatureVariant(
                id="retrieval",
                label="Retrieval",
                integrations=[IntegrationSelection("retrieval")],
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
    )

    assert not job.applicable
    assert job.skip_reason == (
        "integration retrieval requires environment: RETRIEVAL_TOKEN"
    )
    assert "env" not in job.config["agents"][0]
    assert job.config["agents"][0]["extra_allowed_hosts"] == [
        "api.wandb.ai",
        "trace.wandb.ai",
        "host.docker.internal",
        "api.openai.com",
    ]
    assert job.config["fugue"]["integration_ids"] == ["retrieval"]


def test_external_http_integration_renders_endpoint_allowlist_and_provenance(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    integration = tmp_path / "configs" / "fugue" / "integrations" / "api.yaml"
    integration.parent.mkdir(parents=True)
    integration.write_text(
        """
id: api
version: "2026-07-14"
support: supported
runtime: {type: external, url: https://api.example.test/v1}
allowed_hosts: [API.EXAMPLE.TEST]
interfaces:
  - {type: http, name: search, path: /search}
"""
    )
    experiment = ExperimentSpec(
        id="external-api",
        title="External API",
        variants=[
            FeatureVariant(
                id="api",
                label="API",
                integrations=[IntegrationSelection("api")],
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
    )

    agent = job.config["agents"][0]
    assert job.applicable
    assert agent["extra_allowed_hosts"] == [
        "api.wandb.ai",
        "trace.wandb.ai",
        "host.docker.internal",
        "api.example.test",
        "api.openai.com",
    ]
    assert agent["env"] == {
        "FUGUE_INTEGRATION_API_SEARCH_URL": "https://api.example.test/v1/search"
    }
    assert job.integration_provenance[0]["version"] == "2026-07-14"
    assert len(job.integration_provenance[0]["config_hash"]) == 64


def test_integration_instruction_content_not_presentation_or_output_capture_controls_identity(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    root = tmp_path / "configs" / "fugue" / "integrations"
    instructions = root / "review"
    instructions.mkdir(parents=True)
    usage = instructions / "usage.md"
    usage.write_text("Use the first review policy.\n")
    declaration = root / "review.yaml"
    declaration.write_text(
        """
id: review
version: "1"
support: experimental
runtime: {type: builtin, command: [python, -m, example_server]}
interfaces:
  - {type: mcp, name: review, transport: stdio}
instructions: [usage.md]
artifacts:
  - {source: /logs/first-review.jsonl}
"""
    )
    experiment = ExperimentSpec(
        id="instruction-identity",
        title="Instruction identity",
        variants=[
            FeatureVariant(
                id="review",
                label="Review",
                integrations=[IntegrationSelection("review")],
            )
        ],
    )

    def rendered(run_id: str):
        [job] = render_jobs(
            experiment=experiment,
            manifest=load_manifest(manifest_path),
            manifest_path=manifest_path,
            repo_root=tmp_path,
            env={},
            model="openai/gpt-5",
            run_id=run_id,
        )
        return job

    first = rendered("first")
    usage.write_text("Use the second review policy.\n")
    second = rendered("second")
    (instructions / "renamed.md").write_text(usage.read_text())
    declaration.write_text(
        """
id: review
version: "1"
support: supported
runtime: {type: builtin, command: [python, -m, example_server]}
interfaces:
  - {type: mcp, name: review, transport: stdio}
instructions: [renamed.md]
artifacts:
  - source: /logs/renamed-review.jsonl
    destination: review.jsonl
    exclude: ["*.tmp"]
"""
    )
    presentation_only = rendered("presentation")

    assert first.candidate_id != second.candidate_id
    assert second.candidate_id == presentation_only.candidate_id
    assert (
        first.integration_provenance[0]["instruction_assets"][0]["sha256"]
        != second.integration_provenance[0]["instruction_assets"][0]["sha256"]
    )
    assert (
        second.integration_provenance[0]["config_hash"]
        != presentation_only.integration_provenance[0]["config_hash"]
    )
    assert (
        second.resolved_candidate.definition["integrations"]
        == presentation_only.resolved_candidate.definition["integrations"]
    )


def test_stdio_integration_mounts_and_invokes_the_policy_proxy(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    integration = tmp_path / "configs" / "fugue" / "integrations" / "stdio.yaml"
    integration.parent.mkdir(parents=True)
    integration.write_text(
        """
id: stdio
version: "1"
runtime: {type: builtin, command: [python, -m, example_server]}
interfaces:
  - type: mcp
    name: reviewed
    transport: stdio
    allowed_tools: [search]
"""
    )
    experiment = ExperimentSpec(
        id="stdio",
        title="Stdio",
        variants=[
            FeatureVariant(
                id="stdio",
                label="Stdio",
                integrations=[IntegrationSelection("stdio")],
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
    )

    agent = job.config["agents"][0]
    [server] = agent["mcp_servers"]
    assert server["command"] == "python"
    assert server["args"][:5] == [
        "-m",
        "fugue.mcp_proxy",
        "--name",
        "reviewed",
        "--allow-tool",
    ]
    assert "search" in server["args"]
    assert server["args"][-4:] == ["--", "python", "-m", "example_server"]
    assert agent["env"]["PYTHONPATH"] == "/fugue-src"
    assert any(
        mount["target"] == "/fugue-src/fugue" and mount["read_only"] is True
        for mount in job.config["environment"]["mounts"]
    )


def test_attempts_render_as_independent_comparable_trials(tmp_path: Path):
    manifest_path = tmp_path / "tasks.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        n_attempts=2,
        variants=[FeatureVariant(id="baseline", label="Baseline")],
    )

    jobs = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
    )

    assert [job.trial_index for job in jobs] == [1, 2]
    assert len({job.comparison_example_id for job in jobs}) == 1
    assert len({job.candidate_id for job in jobs}) == 1
    assert all(job.config["n_attempts"] == 1 for job in jobs)
    assert jobs[0].job_name.endswith("-t001")
    assert jobs[1].job_name.endswith("-t002")


def test_presentation_and_scoring_do_not_change_candidate_identity(tmp_path: Path):
    manifest_path = tmp_path / "tasks.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="presentation-only",
        title="Presentation only",
        variants=[
            FeatureVariant(id="first-name", label="First label"),
            FeatureVariant(
                id="renamed",
                label="Different label",
                verifier={"type": "pytest"},
            ),
        ],
    )

    jobs = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
    )

    assert len(jobs) == 2
    assert jobs[0].candidate_id == jobs[1].candidate_id
    assert (
        jobs[0].resolved_candidate.definition == jobs[1].resolved_candidate.definition
    )
    assert (
        jobs[0].resolved_candidate.execution_fingerprint
        != jobs[1].resolved_candidate.execution_fingerprint
    )


def test_missing_context_capability_is_not_applicable(tmp_path: Path):
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        variants=[
            FeatureVariant(
                id="agentsmd",
                label="AGENTS",
                context=ContextSelection(system_id="agentsmd"),
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
        required_capabilities=["retrieve"],
    )

    assert job.applicable is False
    assert job.skip_reason == "missing context capabilities: retrieve"
    assert job.config["fugue"]["applicable"] is False


def test_portable_context_binding_uses_pinned_sidecar_and_command(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        "fugue.bench.job_config.read_portable_runtime_lock",
        lambda root: {
            "image": "fugue-context-runtime:locked",
            "image_id": "sha256:" + "1" * 64,
            "recipe_sha256": "2" * 64,
        },
    )
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        variants=[
            FeatureVariant(
                id="rag",
                label="RAG",
                context=ContextSelection(system_id="rag-bm25"),
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
    )

    agent = job.config["agents"][0]
    assert "mcp_servers" not in agent
    assert agent["env"]["FUGUE_CONTEXT_COMMAND"] == "fugue-context"
    assert agent["env"]["FUGUE_CONTEXT_QUERY_URL"] == "http://127.0.0.1:8001"
    assert job.context_delivery == "portable"
    assert job.config["fugue"]["context_delivery"] == "portable"
    assert job.config["fugue"]["context_runtime_required"] is True
    assert job.env["FUGUE_CONTEXT_DELIVERY"] == "portable"
    compose_paths = job.config["environment"]["extra_docker_compose"]
    compose_path = next(path for path in compose_paths if "context-runtime" in path)
    assert job.generated_runtime_files == tuple(
        tmp_path / path for path in compose_paths
    )
    policy_path = next(path for path in compose_paths if "trial-policy" in path)
    policy = yaml.safe_load((tmp_path / policy_path).read_text())
    assert policy["services"]["main"]["pull_policy"] == "never"
    descriptor = job.resolved_candidate.execution_definition["context_runtime"]
    assert descriptor == {
        "bridge_url": "http://host.docker.internal:4000",
        "image": "sha256:" + "1" * 64,
        "image_id": "sha256:" + "1" * 64,
        "kind": "compose_service",
        "mcp_port": 8000,
        "network": "shared_main_namespace",
        "portable_port": 8001,
        "prepared": True,
        "query_url": "http://127.0.0.1:8001",
        "recipe_sha256": "2" * 64,
        "schema_version": 1,
        "service": "fugue-context",
    }
    compose_text = (tmp_path / compose_path).read_text()
    assert "secret" not in compose_text
    compose = yaml.safe_load(compose_text)
    service = compose["services"]["fugue-context"]
    assert service["image"] == descriptor["image"]
    assert "build" not in service
    assert service["pull_policy"] == "never"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "extra_hosts" not in service
    assert service["environment"]["FUGUE_BRIDGE_BASE_URL"] == descriptor["bridge_url"]
    assert "WANDB_API_KEY" not in service["environment"]
    assert agent["env"]["FUGUE_CONTEXT_QUERY_URL"] == descriptor["query_url"]
    dockerfile = Path(__file__).parents[1] / "Dockerfile.context"
    assert "ghcr.io/astral-sh/uv:0.11.27" in dockerfile.read_text()
    assert "fugue.context_server" in service["command"]
    assert "8001" in service["healthcheck"]["test"][-1]
    assert (
        next(iter(job.context_cache_keys.values())) in service["volumes"][0]["source"]
    )
    assert service["network_mode"] == "service:main"
    assert "main" not in compose["services"]
    client_mount = next(
        item
        for item in job.config["environment"]["mounts"]
        if item["target"] == "/usr/local/bin/fugue-context"
    )
    assert client_mount["read_only"] is True
    assert "context_client.py" in client_mount["source"]
    assert not any(
        item["target"] == "/fugue-context"
        for item in job.config["environment"]["mounts"]
    )
    assert "artifacts" not in job.config


def test_portable_context_does_not_inject_native_mcp_or_collide(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    integration = tmp_path / "configs" / "fugue" / "integrations" / "api.yaml"
    integration.parent.mkdir(parents=True)
    integration.write_text(
        """
id: api
version: "1"
runtime:
  type: compose
  image: ghcr.io/example/api@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  service: api
  port: 8000
interfaces:
  - {type: http, name: api, path: /}
"""
    )
    experiment = ExperimentSpec(
        id="port-collision",
        title="Port collision",
        variants=[
            FeatureVariant(
                id="combined",
                label="Combined",
                context=ContextSelection(system_id="rag-bm25"),
                integrations=[IntegrationSelection("api")],
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
    )

    assert job.applicable is True
    assert "mcp_servers" not in job.config["agents"][0]
    assert job.config["agents"][0]["env"]["FUGUE_CONTEXT_COMMAND"] == ("fugue-context")
    compose_paths = [
        tmp_path / path for path in job.config["environment"]["extra_docker_compose"]
    ]
    context_path = next(
        path for path in compose_paths if "context-runtime" in path.as_posix()
    )
    integration_path = next(
        path for path in compose_paths if "integrations" in path.as_posix()
    )
    assert job.generated_runtime_files == tuple(compose_paths)
    context_service = yaml.safe_load(context_path.read_text())["services"][
        "fugue-context"
    ]
    integration_service = yaml.safe_load(integration_path.read_text())["services"][
        "api"
    ]
    assert context_service["network_mode"] == "service:main"
    assert integration_service["network_mode"] == "service:main"


def test_bridged_codex_native_mcp_is_registered_outside_model_route(tmp_path: Path):
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        variants=[
            FeatureVariant(
                id="rag",
                label="RAG",
                context=ContextSelection(system_id="rag-bm25", delivery="native_mcp"),
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="wandb/zai-org/GLM-5.2",
        run_id="unit",
    )

    agent = job.config["agents"][0]
    [server] = agent["mcp_servers"]
    assert server["name"] == "fugue-context"
    assert server["transport"] == "stdio"
    assert "fugue.context_server" in server["args"]
    [policy_path] = job.config["environment"]["extra_docker_compose"]
    policy = yaml.safe_load((tmp_path / policy_path).read_text())
    assert policy["services"]["main"]["pull_policy"] == "never"
    assert any(
        mount["target"] == "/fugue-context"
        for mount in job.config["environment"]["mounts"]
    )
    assert "artifacts" not in job.config
    assert job.applicable is True
    assert job.skip_reason is None
    assert job.config["fugue"]["model_provider"] == "wandb"
    assert job.config["fugue"]["harness_capabilities"] == {
        "native_mcp": True,
        "isolated_home": True,
        "provider_independent_tools": True,
    }


def test_bridged_codex_portable_context_is_applicable(tmp_path: Path):
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        variants=[
            FeatureVariant(
                id="rag",
                label="RAG",
                context=ContextSelection(system_id="rag-bm25"),
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="wandb/zai-org/GLM-5.2",
        run_id="unit",
    )

    assert job.applicable is True
    assert "mcp_servers" not in job.config["agents"][0]
    assert job.config["agents"][0]["env"]["FUGUE_CONTEXT_COMMAND"] == ("fugue-context")


def test_unreviewed_harness_native_mcp_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: custom, agent: example.agent:Custom}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        variants=[
            FeatureVariant(
                id="rag",
                label="RAG",
                context=ContextSelection(system_id="rag-bm25", delivery="native_mcp"),
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="wandb/zai-org/GLM-5.2",
        run_id="unit",
    )

    assert job.applicable is False
    assert job.skip_reason == (
        "harness adapter example.agent:Custom has no reviewed native MCP "
        "registration contract"
    )


def test_openai_codex_native_mcp_is_eligible_for_runtime_probe(tmp_path: Path):
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        variants=[
            FeatureVariant(
                id="rag",
                label="RAG",
                context=ContextSelection(system_id="rag-bm25", delivery="native_mcp"),
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
    )

    assert job.applicable is True
    assert job.context_delivery == "native_mcp"
    [server] = job.config["agents"][0]["mcp_servers"]
    assert server["name"] == "fugue-context"
    assert server["transport"] == "stdio"
    assert "fugue.context_server" in server["args"]


def test_portable_context_contract_is_identical_for_all_harnesses(tmp_path: Path):
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: hermes, agent: fugue.agents:FugueHermes}
  - {name: openclaw, agent: fugue.agents:FugueOpenClaw}
  - {name: claude-code, agent: fugue.agents:FugueClaudeCode}
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        variants=[
            FeatureVariant(
                id="rag",
                label="RAG",
                context=ContextSelection(system_id="rag-bm25"),
            )
        ],
    )

    jobs = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={"WANDB_API_KEY": "must-not-serialize"},
        model="wandb/zai-org/GLM-5.2",
        run_id="unit",
    )

    assert {job.harness for job in jobs} == {
        "hermes",
        "openclaw",
        "claude-code",
        "codex",
    }
    assert all(job.applicable for job in jobs)
    for job in jobs:
        agent = job.config["agents"][0]
        assert agent["env"]["FUGUE_CONTEXT_COMMAND"] == "fugue-context"
        assert agent["env"]["FUGUE_CONTEXT_QUERY_URL"] == "http://127.0.0.1:8001"
        assert "mcp_servers" not in agent
        assert "must-not-serialize" not in json.dumps(job.config)


def test_external_mcp_without_pinned_runtime_is_not_applicable(tmp_path: Path):
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        variants=[
            FeatureVariant(
                id="codegraph",
                label="CodeGraph",
                context=ContextSelection(system_id="codegraph"),
            )
        ],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
    )

    assert job.applicable is False
    assert (
        job.skip_reason == "context system codegraph does not support portable delivery"
    )


def test_metadata_trace_policy_rejects_unsupported_harness(tmp_path: Path):
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: claude-code, agent: fugue.agents:FugueClaudeCode}
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        trace_content="metadata",
        variants=[FeatureVariant(id="baseline", label="Baseline")],
    )

    jobs = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
    )

    claude = next(job for job in jobs if job.harness == "claude-code")
    codex = next(job for job in jobs if job.harness == "codex")
    assert claude.applicable is False
    assert "cannot guarantee metadata-only" in str(claude.skip_reason)
    assert codex.applicable is True
    assert codex.env["FUGUE_TRACE_CONTENT"] == "metadata"
    assert codex.config["fugue"]["trace_content"] == "metadata"


def test_context_workload_renders_one_portable_binding_per_task(tmp_path: Path):
    manifest_path = tmp_path / "tasks.yaml"
    manifest_path.write_text(
        """
dataset: {ref: fixture/tasks}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-a, repository: {type: git, url: https://github.com/fixture/a, commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}}
  - {id: task-b, repository: {type: git, url: https://github.com/fixture/b, commit: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb}}
"""
    )
    experiment = ExperimentSpec(
        id="experiment-a",
        title="Experiment A",
        variants=[
            FeatureVariant(
                id="rag",
                label="RAG",
                context=ContextSelection(system_id="rag-bm25"),
            )
        ],
    )

    jobs = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="openai/gpt-5",
        run_id="unit",
        required_capabilities=["bind"],
    )

    assert len(jobs) == 2
    assert {tuple(job.context_cache_keys) for job in jobs} == {("task-a",), ("task-b",)}
    for job in jobs:
        [dataset] = job.config["datasets"]
        assert dataset["task_names"] == [
            f"fixture/{task_id}" for task_id in job.context_cache_keys
        ]
        mounts = job.config["environment"]["mounts"]
        client_mount = next(
            item for item in mounts if item["target"] == "/usr/local/bin/fugue-context"
        )
        assert client_mount["read_only"] is True
        compose_path = next(
            path
            for path in job.config["environment"]["extra_docker_compose"]
            if "context-runtime" in path
        )
        compose = yaml.safe_load((tmp_path / compose_path).read_text())
        sidecar_mount = compose["services"]["fugue-context"]["volumes"][0]
        assert next(iter(job.context_cache_keys.values())) in sidecar_mount["source"]
