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


def test_fugue_mcp_binding_uses_pinned_context_sidecar(tmp_path: Path):
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

    [server] = job.config["agents"][0]["mcp_servers"]
    assert server == {
        "name": "fugue-context",
        "transport": "streamable-http",
        "url": "http://fugue-context:8000/mcp",
    }
    [compose_path] = job.config["environment"]["extra_docker_compose"]
    compose_text = (tmp_path / compose_path).read_text()
    assert "secret" not in compose_text
    compose = yaml.safe_load(compose_text)
    service = compose["services"]["fugue-context"]
    assert service["image"] == "fugue-context-runtime:0.1.0"
    assert service["build"]["dockerfile"] == "Dockerfile.context"
    assert "fugue.context_server" in service["command"]
    assert next(iter(job.context_cache_keys.values())) in service["volumes"][0]["source"]
    assert compose["services"]["main"]["depends_on"]["fugue-context"] == {
        "condition": "service_healthy"
    }
    assert {
        "source": "/logs/artifacts/fugue-context-events.jsonl",
        "destination": "fugue-context-events.jsonl",
        "service": "fugue-context",
    } in job.config["artifacts"]


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


def test_context_workload_renders_one_native_binding_per_task(tmp_path: Path):
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
        assert dataset["task_names"] == list(job.context_cache_keys)
        mounts = job.config["environment"]["mounts"]
        context_mount = next(item for item in mounts if item["target"] == "/fugue-context")
        assert next(iter(job.context_cache_keys.values())) in context_mount["source"]
        [compose_path] = job.config["environment"]["extra_docker_compose"]
        compose = yaml.safe_load((tmp_path / compose_path).read_text())
        sidecar_mount = compose["services"]["fugue-context"]["volumes"][0]
        assert next(iter(job.context_cache_keys.values())) in sidecar_mount["source"]
