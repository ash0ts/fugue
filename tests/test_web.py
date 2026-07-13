from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest


def write_pilot_manifest(tmp_path: Path) -> None:
    repo_manifest = tmp_path / "datasets"
    repo_manifest.mkdir(exist_ok=True)
    (repo_manifest / "pilot.yaml").write_text(
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


def make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    write_pilot_manifest(tmp_path)

    from fugue.web import create_app

    return ASGIClient(create_app())


class ASGIClient:
    def __init__(self, app) -> None:
        self.app = app

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs):
        return self.request("PUT", url, **kwargs)

    def request(self, method: str, url: str, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send())


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
    }
    assert "matrix" not in data


def test_web_summary_uses_active_experiment_and_selected_model_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "WANDB_API_KEY=trace-secret\n"
        "WANDB_ENTITY=entity\n"
        "WANDB_PROJECT=project\n"
        "OPENAI_API_KEY=openai-secret\n"
    )
    client = make_client(tmp_path, monkeypatch)
    active = tmp_path / "datasets" / "active.yaml"
    active.write_text(
        """
dataset: {ref: test/dataset}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: one}
  - {id: two}
"""
    )
    client.put(
        "/api/experiments/active",
        json={
            "body": """
id: active
manifest: datasets/active.yaml
model: openai/gpt-5
builder_model: anthropic/claude-sonnet-4-5
variants: [{id: baseline, label: Baseline}]
"""
        },
    )

    response = client.get(
        "/api/summary",
        params={
            "experiment_id": "active",
            "model": "openai/gpt-5",
            "builder_model": "anthropic/claude-sonnet-4-5",
        },
    )

    data = response.json()
    assert data["manifest"]["counts"] == {"tasks": 2, "harnesses": 1}
    assert data["status"]["routes"]["target"]["provider"] == "openai"
    assert data["status"]["routes"]["builder"]["provider"] == "anthropic"
    assert data["readiness"]["model"] is True
    assert data["readiness"]["builder"] is False


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
    assert data["summary"]["by_context_system"][0]["name"] == "rag-bm25"
    assert data["summary"]["by_workload"][0]["name"] == "coding"
    assert data["summary"]["by_run_name"][0]["name"] == "fixture-exp"
    assert data["summary"]["by_experiment_id"][0]["name"] == "fixture-exp-id"
    assert data["summary"]["by_variant_id"][0]["name"] == "baseline"
    assert data["summary"]["by_prompt"][0]["name"] == "smoke-prompt"
    assert data["summary"]["by_skill"][0]["name"] == "repo-skill"
    assert data["summary"]["by_provider"][0]["name"] == "wandb"
    assert data["rows"][0]["weave_url"] == "https://wandb.test/test/fugue/weave"


def test_web_results_count_only_trials_and_mask_context_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "sk-test-secret-value"
    (tmp_path / ".env").write_text(f"OPENAI_API_KEY={secret}\n")
    client = make_client(tmp_path, monkeypatch)
    runtime = tmp_path / ".fugue" / "runtime" / "run-one"
    runtime.mkdir(parents=True)
    (runtime / "context-results.jsonl").write_text(
        json.dumps(
            {
                "record_type": "retrieval",
                "run_id": "run-one",
                "exception_message": f"provider rejected {secret}",
            }
        )
        + "\n"
    )
    jobs = Path(__file__).parent / "fixtures" / "export" / "jobs"

    response = client.get("/api/results", params={"path": jobs.as_posix()})

    assert response.json()["summary"]["total"] == 1
    assert secret not in response.text
    assert "[redacted]" in response.text


def test_static_shell_renders_operator_tabs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = make_client(tmp_path, monkeypatch)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    html = response.text
    assert "Run" in html
    assert "Compare" in html
    assert "Setup" in html
    assert "Feature variants" in html
    assert "Context system" in html
    assert "Trials per cell" in html
    assert "Edit prompt" in html
    assert "Advanced config" in html
    assert 'id="comparisonRows"' in html
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
    context: {system_id: none}
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
    context: {system_id: none}
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
                "variants": [
                    {
                        "id": "baseline",
                        "label": "Baseline",
                        "context": {"system_id": "openwiki"},
                    }
                ],
            },
            "harnesses": ["codex"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    config_path = tmp_path / data["commands"][0]["config_path"]
    assert data["summary"]["cells"] == 1
    assert not config_path.exists()
    assert not (tmp_path / ".fugue" / "runtime" / "web-preview").exists()
    assert not (tmp_path / "configs" / "fugue" / "experiments" / "scratch.yaml").exists()


def test_preview_endpoint_supports_direct_sequence_workloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context_dir = tmp_path / "configs" / "fugue" / "context-systems"
    context_dir.mkdir(parents=True)
    (context_dir / "none.yaml").write_text(
        """
id: none
title: No added context
provider: fugue.bench.context:EmptyContextProvider
version: "1"
capabilities: [prepare, bind, ingest, sequence]
"""
    )
    (tmp_path / "datasets").mkdir(exist_ok=True)
    (tmp_path / "datasets" / "continuity.yaml").write_text(
        """
id: continuity
runner: sequence
sequences:
  - id: one
    repo: example/repo
    commit: abc123
    events:
      - {episode: 1, kind: fact, content: Use focused tests.}
    probes:
      - id: recall
        after_episode: 1
        query: How should this be tested?
        expected_facts: [focused tests]
"""
    )
    client = make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/preview",
        json={
            "experiment_id": "context-preview",
            "experiment": {
                "id": "context-preview",
                "title": "Context preview",
                "workloads": [
                    {
                        "id": "continuity",
                        "runner": "sequence",
                        "dataset": "datasets/continuity.yaml",
                        "required_capabilities": ["sequence"],
                        "systems": ["none"],
                    }
                ],
                "presets": {
                    "smoke": {
                        "workloads": ["continuity"],
                        "systems": ["none"],
                    }
                },
                "default_preset": "smoke",
                "variants": [
                    {
                        "id": "none",
                        "label": "No added context",
                        "context": {"system_id": "none"},
                    }
                ],
            },
            "preset": "smoke",
            "workloads": ["continuity"],
            "systems": ["none"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["cells"] == 1
    assert data["summary"]["estimated_trials"] == 1
    assert data["summary"]["harnesses"] == 0
    assert data["summary"]["direct_runners"] == ["sequence"]
    assert data["commands"][0]["harness"] == "sequence"


def test_context_system_endpoint_reports_capabilities_and_license_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = make_client(tmp_path, monkeypatch)

    response = client.get("/api/context-systems")

    assert response.status_code == 200
    systems = {item["id"]: item for item in response.json()}
    assert "retrieve" in systems["rag-bm25"]["capabilities"]
    assert systems["gitnexus"]["requires_license_approval"] is True
    assert systems["gitnexus"]["license_ready"] is False


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


def test_web_jobs_recover_orphans_and_stream_incrementally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = make_client(tmp_path, monkeypatch)
    from fugue import web_jobs

    orphan = tmp_path / "jobs" / "web" / "orphan"
    orphan.mkdir(parents=True)
    (orphan / "meta.json").write_text(
        json.dumps(
            {
                "id": "orphan",
                "status": "running",
                "owner": "previous-server",
                "pid": os.getpid(),
            }
        )
    )
    assert client.get("/api/jobs/orphan").json()["status"] == "interrupted"

    active = tmp_path / "jobs" / "web" / "active"
    active.mkdir(parents=True)
    meta_path = active / "meta.json"
    meta = {
        "id": "active",
        "status": "running",
        "owner": web_jobs._SERVER_ID,
        "pid": os.getpid(),
    }
    meta_path.write_text(json.dumps(meta))
    (active / "output.log").write_text("first chunk\n")
    events = web_jobs.tail_job_events("active")
    assert "first chunk" in next(events)
    meta["status"] = "succeeded"
    meta_path.write_text(json.dumps(meta))
    assert '"done": true' in next(events)
