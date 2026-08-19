from __future__ import annotations

from pathlib import Path

import pytest

from fugue.bench.job_config import render_jobs
from fugue.bench.library import (
    ExperimentSpec,
    FeatureVariant,
    IntegrationSelection,
)
from fugue.bench.manifest import load_manifest

_HARNESSES = {
    "hermes": "fugue.agents:FugueHermes",
    "openclaw": "fugue.agents:FugueOpenClaw",
    "claude-code": "fugue.agents:FugueClaudeCode",
    "codex": "fugue.agents:FugueCodex",
}


def _manifest(root: Path, harness: str = "codex") -> Path:
    path = root / "pilot.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
dataset: {{ref: fixture/tasks}}
harnesses:
  - {{name: {harness}, agent: {_HARNESSES[harness]}}}
tasks:
  - {{id: task-a}}
""",
        encoding="utf-8",
    )
    return path


def _render(
    root: Path,
    *,
    experiment: ExperimentSpec,
    env: dict[str, str],
    model: str,
    harness: str = "codex",
):
    manifest_path = _manifest(root, harness)
    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=root,
        env=env,
        model=model,
        run_id="credential-isolation",
    )
    return job


@pytest.mark.parametrize("harness", tuple(_HARNESSES))
def test_local_harbor_default_denies_unreviewed_sensitive_names_for_every_harness(
    tmp_path: Path,
    harness: str,
) -> None:
    job = _render(
        tmp_path,
        experiment=ExperimentSpec(
            id=f"local-sensitive-{harness}",
            title="Local sensitive environment isolation",
            evidence_mode="local",
            variants=[FeatureVariant(id="baseline", label="Baseline")],
        ),
        env={
            "ANTHROPIC_API_KEY": "selected-model-key",
            "ANTHROPIC_BASE_URL": "https://selected-anthropic.test",
            "OPENAI_API_KEY": "unselected-model-key",
            "OPENAI_BASE_URL": "https://unselected-openai.test/v1",
            "GITHUB_ACCESS_TOKEN": "ambient-github-token",
            "DATABASE_PASSWORD": "ambient-database-password",
            "DEPLOY_SECRET": "ambient-deploy-secret",
            "CUSTOM_API_KEY": "ambient-custom-key",
            "PUBLIC_SETTING": "kept",
        },
        model="anthropic/claude-sonnet-5",
        harness=harness,
    )

    assert job.env["ANTHROPIC_API_KEY"] == "selected-model-key"
    assert job.env["ANTHROPIC_BASE_URL"] == "https://selected-anthropic.test"
    assert job.env["PUBLIC_SETTING"] == "kept"
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GITHUB_ACCESS_TOKEN",
        "DATABASE_PASSWORD",
        "DEPLOY_SECRET",
        "CUSTOM_API_KEY",
    ):
        assert name not in job.env


def _ambient_wandb_environment() -> dict[str, str]:
    return {
        "WANDB_API_KEY": "ambient-general-key",
        "FUGUE_WEAVE_API_KEY": "ambient-evidence-key",
        "FUGUE_WEAVE_PROJECT": "ambient/evidence",
        "WEAVE_PROJECT": "ambient/evidence",
        "WANDB_ENTITY": "ambient",
        "WANDB_PROJECT": "evidence",
        "WANDB_BASE_URL": "https://api.wandb.test",
        "FUGUE_WEAVE_BASE_URL": "https://api.wandb.test",
        "FUGUE_WEAVE_TRACE_SERVER_URL": "https://trace.wandb.test",
        "WF_TRACE_SERVER_URL": "https://trace.wandb.test",
        "WANDB_APP_BASE_URL": "https://app.wandb.test",
        "FUGUE_SOURCE_EVIDENCE_PROJECT": "ambient/source",
        "FUGUE_WANDB_RESEARCH_ID": "ambient-research",
        "FUGUE_WANDB_STUDY_ID": "ambient-study",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.wandb.test",
        "OTEL_RESOURCE_ATTRIBUTES": "wandb.project=ambient/evidence",
    }


def test_local_harbor_drops_ambient_wandb_and_weave_environment(
    tmp_path: Path,
) -> None:
    env = {
        **_ambient_wandb_environment(),
        "ANTHROPIC_API_KEY": "selected-model-key",
        "OPENAI_API_KEY": "unselected-model-key",
        "PUBLIC_SETTING": "kept",
    }
    job = _render(
        tmp_path,
        experiment=ExperimentSpec(
            id="local-anthropic",
            title="Local Anthropic",
            evidence_mode="local",
            variants=[FeatureVariant(id="baseline", label="Baseline")],
        ),
        env=env,
        model="anthropic/claude-sonnet-5",
    )

    assert job.env["ANTHROPIC_API_KEY"] == "selected-model-key"
    assert "OPENAI_API_KEY" not in job.env
    assert job.env["PUBLIC_SETTING"] == "kept"
    for name in _ambient_wandb_environment():
        assert name not in job.env


def test_local_wandb_model_keeps_only_dedicated_inference_environment(
    tmp_path: Path,
) -> None:
    env = {
        **_ambient_wandb_environment(),
        "ANTHROPIC_API_KEY": "unselected-anthropic-key",
        "OPENAI_API_KEY": "unselected-openai-key",
        "FUGUE_WANDB_INFERENCE_API_KEY": "selected-inference-key",
        "FUGUE_WANDB_INFERENCE_PROJECT": "team/inference",
        "FUGUE_WANDB_INFERENCE_BASE_URL": "https://inference.wandb.test/v1",
    }
    job = _render(
        tmp_path,
        experiment=ExperimentSpec(
            id="local-wandb-model",
            title="Local W&B model",
            evidence_mode="local",
            variants=[FeatureVariant(id="baseline", label="Baseline")],
        ),
        env=env,
        model="wandb/zai-org/GLM-5.2",
    )

    assert job.env["FUGUE_WANDB_INFERENCE_API_KEY"] == "selected-inference-key"
    assert job.env["FUGUE_WANDB_INFERENCE_PROJECT"] == "team/inference"
    assert job.env["FUGUE_WANDB_INFERENCE_BASE_URL"] == (
        "https://inference.wandb.test/v1"
    )
    assert "WANDB_API_KEY" not in job.env
    assert "ANTHROPIC_API_KEY" not in job.env
    assert "OPENAI_API_KEY" not in job.env
    assert "FUGUE_WEAVE_API_KEY" not in job.env
    assert "WANDB_BASE_URL" not in job.env
    assert "FUGUE_WEAVE_PROJECT" not in job.env


def test_local_wandb_model_keeps_legacy_general_key_when_it_is_model_key(
    tmp_path: Path,
) -> None:
    job = _render(
        tmp_path,
        experiment=ExperimentSpec(
            id="local-wandb-legacy",
            title="Local W&B legacy model key",
            evidence_mode="local",
            variants=[FeatureVariant(id="baseline", label="Baseline")],
        ),
        env={"WANDB_API_KEY": "legacy-model-key"},
        model="wandb/zai-org/GLM-5.2",
    )

    assert job.env["WANDB_API_KEY"] == "legacy-model-key"


def test_local_openai_model_keeps_openai_key_and_drops_anthropic_key(
    tmp_path: Path,
) -> None:
    job = _render(
        tmp_path,
        experiment=ExperimentSpec(
            id="local-openai",
            title="Local OpenAI",
            evidence_mode="local",
            variants=[FeatureVariant(id="baseline", label="Baseline")],
        ),
        env={
            "OPENAI_API_KEY": "selected-openai-key",
            "OPENAI_BASE_URL": "https://selected-openai.test/v1",
            "ANTHROPIC_API_KEY": "unselected-anthropic-key",
            "ANTHROPIC_BASE_URL": "https://unselected-anthropic.test",
        },
        model="openai/gpt-5",
    )

    assert job.env["OPENAI_API_KEY"] == "selected-openai-key"
    assert job.env["OPENAI_BASE_URL"] == "https://selected-openai.test/v1"
    assert "ANTHROPIC_API_KEY" not in job.env
    assert "ANTHROPIC_BASE_URL" not in job.env


def test_local_integration_keeps_only_its_reviewed_required_credentials(
    tmp_path: Path,
) -> None:
    integration = (
        tmp_path / "configs" / "fugue" / "integrations" / "wandb-mcp.yaml"
    )
    integration.parent.mkdir(parents=True)
    integration.write_text(
        """
id: wandb-mcp
version: "1"
support: supported
runtime:
  type: builtin
  command: [python, -m, wandb_mcp_server]
interfaces:
  - type: mcp
    name: wandb
    transport: stdio
    allowed_tools: [query_wandb_tool]
required_env: [WANDB_API_KEY, SERVICE_ACCESS_TOKEN]
allowed_hosts: [api.wandb.ai]
""",
        encoding="utf-8",
    )
    env = {
        **_ambient_wandb_environment(),
        "ANTHROPIC_API_KEY": "selected-model-key",
        "OPENAI_API_KEY": "unselected-model-key",
        "SERVICE_ACCESS_TOKEN": "reviewed-integration-token",
        "OTHER_ACCESS_TOKEN": "unreviewed-integration-token",
    }
    job = _render(
        tmp_path,
        experiment=ExperimentSpec(
            id="local-wandb-integration",
            title="Local W&B integration",
            evidence_mode="local",
            source_evidence_project="wandb/locked-source",
            variants=[
                FeatureVariant(
                    id="candidate",
                    label="Candidate",
                    integrations=[IntegrationSelection("wandb-mcp")],
                )
            ],
        ),
        env=env,
        model="anthropic/claude-sonnet-5",
    )

    assert job.env["WANDB_API_KEY"] == "ambient-general-key"
    assert job.config["agents"][0]["env"]["WANDB_API_KEY"] == (
        "${WANDB_API_KEY}"
    )
    assert job.env["SERVICE_ACCESS_TOKEN"] == "reviewed-integration-token"
    assert job.config["agents"][0]["env"]["SERVICE_ACCESS_TOKEN"] == (
        "${SERVICE_ACCESS_TOKEN}"
    )
    assert job.env["FUGUE_SOURCE_EVIDENCE_PROJECT"] == "wandb/locked-source"
    assert "FUGUE_WEAVE_API_KEY" not in job.env
    assert "OPENAI_API_KEY" not in job.env
    assert "OTHER_ACCESS_TOKEN" not in job.env
    assert "WANDB_BASE_URL" not in job.env
    assert "FUGUE_WEAVE_PROJECT" not in job.env


def test_weave_required_harbor_preserves_existing_evidence_environment(
    tmp_path: Path,
) -> None:
    env = {
        **_ambient_wandb_environment(),
        "ANTHROPIC_API_KEY": "selected-model-key",
    }
    job = _render(
        tmp_path,
        experiment=ExperimentSpec(
            id="hosted",
            title="Hosted",
            evidence_mode="weave_required",
            evidence_project="ambient/evidence",
            variants=[FeatureVariant(id="baseline", label="Baseline")],
        ),
        env=env,
        model="anthropic/claude-sonnet-5",
    )

    for name, value in _ambient_wandb_environment().items():
        assert job.env[name] == value
