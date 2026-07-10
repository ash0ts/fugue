from __future__ import annotations

from pathlib import Path

import pytest


def write_pilot_manifest(tmp_path: Path) -> None:
    repo_manifest = tmp_path / "datasets"
    repo_manifest.mkdir()
    (repo_manifest / "pilot.yaml").write_text(
        """
dataset:
  ref: swe-bench/swe-bench-verified
memory_variants:
  - none
  - openwiki
harnesses:
  - name: codex
    agent: fugue.agents:FugueCodex
tasks:
  - id: astropy__astropy-12907
"""
    )


def make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.chdir(tmp_path)
    write_pilot_manifest(tmp_path)

    from fugue.web import create_app

    return TestClient(create_app())


def test_web_status_reports_key_presence_without_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "WANDB_API_KEY=trace-secret\n"
        "WANDB_ENTITY=entity\n"
        "WANDB_PROJECT=project\n"
        "OPENAI_API_KEY=model-secret\n"
        "FUGUE_MODEL=openai/gpt-5\n"
    )
    client = make_client(tmp_path, monkeypatch)

    response = client.get("/api/status")

    assert response.status_code == 200
    text = response.text
    assert "trace-secret" not in text
    assert "model-secret" not in text
    data = response.json()
    assert data["keys"]["WANDB_API_KEY"] is True
    assert data["keys"]["OPENAI_API_KEY"] is True
    assert data["route"]["model"] == "openai/gpt-5"
    assert data["weave_project"] == "entity/project"
    assert data["wandb_project_url"] == "https://wandb.ai/entity/project"
    assert data["weave_project_url"] == "https://wandb.ai/entity/project/weave"
    assert data["wandb_app_base_url"] == "https://wandb.ai"


def test_web_summary_reports_readiness_and_manifest_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "WANDB_API_KEY=trace-secret\n"
        "WANDB_ENTITY=entity\n"
        "WANDB_PROJECT=project\n"
        "FUGUE_MODEL=wandb/zai-org/GLM-5.2\n"
    )
    client = make_client(tmp_path, monkeypatch)

    response = client.get("/api/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["readiness"]["trace"] is True
    assert data["manifest"]["counts"] == {
        "tasks": 1,
        "harnesses": 1,
        "memory_variants": 2,
    }
    assert data["manifest"]["memory_variants"] == ["none", "openwiki"]
    assert "matrix" not in data


def test_web_results_groups_rows_and_adds_weave_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("WANDB_APP_BASE_URL=https://wandb.test\n")
    client = make_client(tmp_path, monkeypatch)
    jobs = Path(__file__).parent / "fixtures" / "export" / "jobs"

    response = client.get("/api/results", params={"path": jobs.as_posix()})

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["total"] == 1
    assert data["summary"]["by_harness"][0]["name"] == "hermes"
    assert data["summary"]["by_feature_memory"][0]["name"] == "none"
    assert data["summary"]["by_run_name"][0]["name"] == "fixture-exp"
    assert data["summary"]["by_experiment_id"][0]["name"] == "fixture-exp-id"
    assert data["summary"]["by_variant_id"][0]["name"] == "baseline"
    assert data["summary"]["by_prompt"][0]["name"] == "smoke-prompt"
    assert data["summary"]["by_skill"][0]["name"] == "repo-skill"
    assert data["summary"]["by_provider"][0]["name"] == "wandb"
    assert data["rows"][0]["weave_url"] == "https://wandb.test/test/fugue/weave"


def test_static_shell_renders_operator_tabs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = make_client(tmp_path, monkeypatch)

    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "Run" in html
    assert "Compare" in html
    assert "Setup" in html
    assert "Feature variants" in html
    assert "Trials per cell" in html
    assert "Edit prompt" in html
    assert "Advanced config" in html
    assert "Open W&B" in html
    assert "Library" not in html
    assert "Profiles" not in html
    assert "matrixStave" not in html


def test_library_endpoints_save_and_reload_repo_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = make_client(tmp_path, monkeypatch)

    response = client.put(
        "/api/prompts/test-prompt",
        json={"body": "# Test prompt\n\nRead the repo first.\n"},
    )
    assert response.status_code == 200
    assert (tmp_path / "configs" / "fugue" / "prompts" / "test-prompt.md").is_file()

    response = client.put(
        "/api/skills/test-skill",
        json={"body": "# Test skill\n\nUse existing patterns.\n"},
    )
    assert response.status_code == 200
    assert (
        tmp_path
        / "configs"
        / "fugue"
        / "skills"
        / "test-skill"
        / "SKILL.md"
    ).is_file()

    response = client.put(
        "/api/experiments/test-experiment",
        json={
            "body": """
id: test-experiment
title: Test experiment
manifest: datasets/pilot.yaml
variants:
  - id: prompt-skill
    label: Prompt plus skill
    prompt_id: test-prompt
    skill_ids: [test-skill]
    memory: none
"""
        },
    )
    assert response.status_code == 200

    response = client.get("/api/library")
    data = response.json()
    assert data["prompts"][0]["id"] == "test-prompt"
    assert data["skills"][0]["id"] == "test-skill"
    assert data["experiments"][0]["id"] == "test-experiment"


def test_render_endpoint_creates_harbor_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = make_client(tmp_path, monkeypatch)
    client.put("/api/prompts/test-prompt", json={"body": "# Test prompt\n"})
    client.put(
        "/api/skills/test-skill",
        json={"body": "# Test skill\n"},
    )
    client.put(
        "/api/experiments/test-experiment",
        json={
            "body": """
id: test-experiment
title: Test experiment
manifest: datasets/pilot.yaml
n_attempts: 2
variants:
  - id: prompt-skill
    label: Prompt plus skill
    prompt_id: test-prompt
    skill_ids: [test-skill]
    memory: none
"""
        },
    )

    response = client.post(
        "/api/render",
        json={
            "experiment_id": "test-experiment",
            "harnesses": ["codex"],
            "n_tasks": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["cells"] == 1
    assert data["summary"]["estimated_trials"] == 2
    assert data["commands"][0]["command"][:3] == ["harbor", "run", "--config"]
    assert data["commands"][0]["variant_id"] == "prompt-skill"
    assert data["commands"][0]["prompt_id"] == "test-prompt"
    assert data["commands"][0]["skill_ids"] == ["test-skill"]
    assert data["commands"][0]["config"]["extra_instruction_paths"] == [
        "configs/fugue/prompts/test-prompt.md"
    ]


def test_preview_endpoint_does_not_write_runtime_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/preview",
        json={
            "experiment_id": "scratch",
            "experiment": {
                "id": "scratch",
                "title": "Scratch",
                "manifest": "datasets/pilot.yaml",
                "variants": [{"id": "baseline", "label": "Baseline", "memory": "openwiki"}],
            },
            "harnesses": ["codex"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    config_path = tmp_path / data["commands"][0]["config_path"]
    assert data["summary"]["cells"] == 1
    assert not config_path.exists()
    assert not (tmp_path / "artifacts" / "memory" / "openwiki" / "INSTRUCTION.md").exists()
    assert not (tmp_path / "configs" / "fugue" / "experiments" / "scratch.yaml").exists()


def test_web_job_detail_returns_metadata_and_log_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = make_client(tmp_path, monkeypatch)
    job_dir = tmp_path / "jobs" / "web" / "job-1"
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(
        '{"id": "job-1", "kind": "run", "status": "succeeded", "command": ["fugue", "run"]}'
    )
    (job_dir / "output.log").write_text("ok\n")

    response = client.get("/api/jobs/job-1")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "job-1"
    assert data["log_tail"] == "ok\n"
