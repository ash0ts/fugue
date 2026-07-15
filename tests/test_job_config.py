from __future__ import annotations

import json
from pathlib import Path

import yaml

from fugue.bench.job_config import render_jobs
from fugue.bench.library import (
    ContextSelection,
    ExperimentSpec,
    FeatureVariant,
    save_prompt,
    save_skill,
)
from fugue.bench.manifest import load_manifest


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
                skill_ids=["skill-a"],
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
    assert instruction_paths[0] == ".fugue/runtime/unit/context-instructions/agentsmd.md"
    assert instruction_paths[-1] == "configs/fugue/prompts/prompt-a.md"
    assert any(path.endswith("/artifact/AGENTS.md") for path in instruction_paths)
    assert config["environment"]["override_memory_mb"] == 4096
    assert config["environment"]["type"] == "docker"
    assert config["artifacts"] == ["/logs/artifacts"]
    assert any(
        mount["target"] == "/fugue-context"
        for mount in config["environment"]["mounts"]
    )
    assert config["verifier"] == {"type": "pytest"}
    assert config["fugue"]["experiment_id"] == "experiment-a"
    assert config["fugue"]["variant_id"] == "prompt-skill"
    assert config["fugue"]["prompt_id"] == "prompt-a"
    assert config["fugue"]["context_system_id"] == "agentsmd"
    assert config["fugue"]["context_version"] == "1"
    assert len(config["fugue"]["context_config_hash"]) == 64
    assert config["fugue"]["agent_config_hash"] == job.agent_config_hash
    assert config["n_attempts"] == 1
    assert config["fugue"]["trial_index"] == 1
    assert config["fugue"]["comparison_example_id"] == job.comparison_example_id
    assert config["fugue"]["expected_evidence_paths"] == {
        "astropy__astropy-12907": ["astropy/modeling/separable.py"]
    }
    assert json.loads(job.env["FUGUE_EXPECTED_EVIDENCE_PATHS"]) == {
        "astropy__astropy-12907": ["astropy/modeling/separable.py"]
    }
    assert config["fugue"]["candidate_id"] == job.candidate_id
    assert config["fugue"]["trace_content"] == "full"
    assert job.env["FUGUE_TRACE_CONTENT"] == "full"


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


def test_portable_context_binding_uses_pinned_sidecar_and_command(tmp_path: Path):
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
    assert agent["env"]["FUGUE_CONTEXT_QUERY_URL"] == (
        "http://fugue-context:8001"
    )
    assert job.context_transport == "portable"
    assert job.config["fugue"]["context_transport"] == "portable"
    assert job.env["FUGUE_CONTEXT_TRANSPORT"] == "portable"
    [compose_path] = job.config["environment"]["extra_docker_compose"]
    compose_text = (tmp_path / compose_path).read_text()
    assert "secret" not in compose_text
    compose = yaml.safe_load(compose_text)
    service = compose["services"]["fugue-context"]
    assert service["image"] == "fugue-context-runtime:0.1.0"
    assert service["build"]["dockerfile"] == "Dockerfile.context"
    dockerfile = Path(__file__).parents[1] / "Dockerfile.context"
    assert "ghcr.io/astral-sh/uv:0.11.27" in dockerfile.read_text()
    assert "fugue.context_server" in service["command"]
    assert "8001" in service["healthcheck"]["test"][-1]
    assert next(iter(job.context_cache_keys.values())) in service["volumes"][0]["source"]
    assert compose["services"]["main"]["depends_on"]["fugue-context"] == {
        "condition": "service_healthy"
    }
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


def test_bridged_codex_native_mcp_is_explicitly_not_applicable(tmp_path: Path):
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
                context=ContextSelection(
                    system_id="rag-bm25", transport="native_mcp"
                ),
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
    assert agent["mcp_servers"] == [
        {
            "name": "fugue-context",
            "transport": "streamable-http",
            "url": "http://fugue-context:8000/mcp",
        }
    ]
    assert job.config["environment"]["extra_docker_compose"]
    assert any(
        mount["target"] == "/fugue-context"
        for mount in job.config["environment"]["mounts"]
    )
    assert "artifacts" not in job.config
    assert job.applicable is False
    assert job.skip_reason == (
        "Codex MCP tools require Responses namespace support; "
        "the wandb bridge accepts function tools only"
    )


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
    assert job.config["agents"][0]["env"]["FUGUE_CONTEXT_COMMAND"] == (
        "fugue-context"
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
                context=ContextSelection(
                    system_id="rag-bm25", transport="native_mcp"
                ),
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
    assert job.context_transport == "native_mcp"
    assert job.config["agents"][0]["mcp_servers"] == [
        {
            "name": "fugue-context",
            "transport": "streamable-http",
            "url": "http://fugue-context:8000/mcp",
        }
    ]


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
        assert agent["env"]["FUGUE_CONTEXT_QUERY_URL"] == (
            "http://fugue-context:8001"
        )
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
    assert job.skip_reason == (
        "runtime:image: provider MCP binding has no pinned runtime_image"
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
  - {id: task-a, repo: fixture/a, base_commit: abc}
  - {id: task-b, repo: fixture/b, base_commit: def}
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
        [compose_path] = job.config["environment"]["extra_docker_compose"]
        compose = yaml.safe_load((tmp_path / compose_path).read_text())
        sidecar_mount = compose["services"]["fugue-context"]["volumes"][0]
        assert next(iter(job.context_cache_keys.values())) in sidecar_mount["source"]
