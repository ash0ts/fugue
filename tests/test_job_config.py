from __future__ import annotations

import json
from pathlib import Path

from fugue.bench.job_config import render_jobs
from fugue.bench.library import ExperimentSpec, FeatureVariant, save_prompt, save_skill
from fugue.bench.manifest import load_manifest


def test_render_jobs_writes_harbor_config_without_secrets(tmp_path: Path):
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset:
  ref: swe-bench/swe-bench-verified
memory_variants: [none]
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
            FeatureVariant(id="baseline", label="Baseline", memory="none"),
            FeatureVariant(
                id="prompt-skill",
                label="Prompt + skill",
                prompt_id="prompt-a",
                skill_ids=["skill-a"],
                memory="agentsmd",
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
    assert job.feature_memory == "agentsmd"
    assert len(job.agent_config_hash) == 64
    config_text = job.config_path.read_text()
    assert "secret-openai" not in config_text
    assert "secret-wandb" not in config_text
    config = json.loads(config_text)
    assert config["agents"][0]["model_name"] == "openai/gpt-5"
    assert config["agents"][0]["skills"] == ["configs/fugue/skills/skill-a"]
    assert config["agents"][0]["kwargs"] == {"temperature": 0}
    assert config["agents"][0]["env"] == {"FUGUE_AGENT_MODE": "strict"}
    assert config["extra_instruction_paths"] == [
        "artifacts/memory/agentsmd/INSTRUCTION.md",
        "configs/fugue/prompts/prompt-a.md",
    ]
    assert config["environment"]["override_memory_mb"] == 4096
    assert config["environment"]["type"] == "docker"
    assert config["verifier"] == {"type": "pytest"}
    assert config["fugue"]["experiment_id"] == "experiment-a"
    assert config["fugue"]["variant_id"] == "prompt-skill"
    assert config["fugue"]["prompt_id"] == "prompt-a"
    assert config["fugue"]["feature_memory"] == "agentsmd"
    assert config["fugue"]["agent_config_hash"] == job.agent_config_hash
