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
conditions:
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
        "conditions": 2,
        "matrix_cells": 2,
    }
    assert data["matrix"][0]["harness"] == "codex"
    assert data["matrix"][0]["cells"][0]["status"] == "ready"


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
    assert data["summary"]["by_condition"][0]["name"] == "none"
    assert data["summary"]["by_provider"][0]["name"] == "wandb"
    assert data["rows"][0]["weave_url"] == "https://wandb.test/test/fugue/weave"


def test_static_shell_renders_operator_tabs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = make_client(tmp_path, monkeypatch)

    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "Overview" in html
    assert "Setup" in html
    assert "Run matrix" in html
    assert "Jobs" in html
    assert "Results" in html
    assert "matrixStave" in html
    assert "Open W&B" in html


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
