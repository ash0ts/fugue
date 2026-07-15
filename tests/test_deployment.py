from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from test_operator import make_operator_repo

from fugue.bench import deployment
from fugue.bench.deployment import (
    _deployment_candidate,
    package_candidate,
    write_run_input_lock,
)
from fugue.bench.execution import plan_cells, write_run_manifest
from fugue.bench.operator import ExperimentRequest


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


def _workspace(path: Path, *, secret: bool = False) -> Path:
    path.mkdir()
    (path / ".gitignore").write_text("ignored.txt\n")
    (path / "app.py").write_text("print('production')\n")
    (path / "ignored.txt").write_text("not packaged\n")
    if secret:
        (path / ".env").write_text("API_KEY=tracked-secret\n")
    _git(path, "init", "-q")
    _git(path, "add", ".")
    _git(
        path,
        "-c",
        "user.name=Fugue Tests",
        "-c",
        "user.email=fugue@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    return path


def _packaging_run(tmp_path: Path, *, run_status: str = "failed"):
    repo = tmp_path / "fugue"
    service = make_operator_repo(repo)
    shutil.copyfile(Path(__file__).parents[1] / "pyproject.toml", repo / "pyproject.toml")
    shutil.copyfile(Path(__file__).parents[1] / "uv.lock", repo / "uv.lock")
    (repo / "README.md").write_text("# Fixture\n")
    shutil.copytree(Path(__file__).parents[1] / "fugue", repo / "fugue")
    experiment = service.experiment("demo")
    variant = replace(
        experiment.variants[0],
        prompt_id="demo-prompt",
        skill_ids=["demo-skill"],
        agent_env={
            "CUSTOM_TOKEN": "trace-secret-value",
            "SHORT_SECRET": "x",
        },
    )
    experiment = replace(experiment, variants=[variant])
    run_id = "run-package"
    jobs = service.rendered_jobs(
        ExperimentRequest(experiment_id="demo"),
        run_id=run_id,
        experiment=experiment,
    )
    write_run_input_lock(
        repo,
        run_id,
        experiment,
        jobs,
        env={"CUSTOM_TOKEN": "trace-secret-value", "SHORT_SECRET": "x"},
    )
    write_run_manifest(
        repo,
        run_id,
        {
            "status": run_status,
            "run_name": "package fixture",
            "experiment_id": "demo",
        },
    )
    [cell] = plan_cells(jobs, run_id=run_id, run_name="package fixture")
    cells_path = repo / ".fugue/runtime" / run_id / "cells.jsonl"
    cells_path.write_text(json.dumps(cell.record("failed")) + "\n")
    return repo, run_id, jobs[0].candidate_id


def test_packages_explicit_imperfect_candidate_reproducibly(tmp_path: Path) -> None:
    repo, run_id, candidate_id = _packaging_run(tmp_path)
    workspace = _workspace(tmp_path / "workspace")

    first = package_candidate(
        repo_root=repo,
        run_id=run_id,
        candidate_id=candidate_id,
        workspace=workspace,
        image="example/fugue:test",
        build=False,
    )
    second = package_candidate(
        repo_root=repo,
        run_id=run_id,
        candidate_id=candidate_id,
        workspace=workspace,
        image="example/fugue:test",
        build=False,
    )

    assert first.deployment_id == second.deployment_id
    assert first.workspace_digest == second.workspace_digest
    assert (first.path / "workspace/app.py").is_file()
    assert not (first.path / "workspace/ignored.txt").exists()
    spec = json.loads(first.spec_path.read_text())
    assert "trace-secret-value" not in (
        repo / ".fugue/runtime" / run_id / "input-lock.json"
    ).read_text()
    assert '"SHORT_SECRET": "${SHORT_SECRET}"' in (
        repo / ".fugue/runtime" / run_id / "input-lock.json"
    ).read_text()
    assert "trace-secret-value" not in first.spec_path.read_text()
    assert spec["candidate_id"] == candidate_id
    assert spec["resources"] == {
        "cpus": 2,
        "memory_mb": 4096,
        "storage_mb": 10240,
        "timeout_sec": 900,
    }
    assert spec["runtime_versions"] == {
        "python": "3.13",
        "harbor": "0.18.0",
        "uv": "0.11.27",
        "fugue": "0.1.0",
    }
    assert spec["protocol_versions"]["open-responses"] == "2026-04-24"
    assert spec["provenance"]["workspace"]["digest"] == first.workspace_digest
    assert spec["candidate"]["model_route"]["responses_base_url"] == (
        "https://api.openai.com/v1"
    )
    dockerfile = (first.path / "Dockerfile").read_text()
    assert "io.fugue.candidate.id" in dockerfile
    assert "io.fugue.input-lock.digest" in dockerfile
    assert "io.fugue.runtime.digest" in dockerfile
    assert "python:3.13-slim" in dockerfile
    assert "uv==0.11.27" in dockerfile
    assert "uv sync --frozen --no-dev --extra serve" in dockerfile


def test_packaging_fails_closed_without_lock_or_with_changed_assets(
    tmp_path: Path,
) -> None:
    repo, run_id, candidate_id = _packaging_run(tmp_path)
    workspace = _workspace(tmp_path / "workspace")
    lock_path = repo / ".fugue/runtime" / run_id / "input-lock.json"
    lock_body = lock_path.read_text()
    lock_path.unlink()
    with pytest.raises(ValueError, match="input lock is missing"):
        package_candidate(
            repo_root=repo,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace=workspace,
            image="example/fugue:test",
            build=False,
        )
    lock_path.write_text(lock_body)
    (repo / "configs/fugue/prompts/demo-prompt.md").write_text("# Changed\n")
    with pytest.raises(ValueError, match="prompt asset changed"):
        package_candidate(
            repo_root=repo,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace=workspace,
            image="example/fugue:test",
            build=False,
        )


def test_packaging_rejects_dirty_or_secret_bearing_workspace(tmp_path: Path) -> None:
    repo, run_id, candidate_id = _packaging_run(tmp_path)
    dirty = _workspace(tmp_path / "dirty")
    (dirty / "app.py").write_text("dirty\n")
    with pytest.raises(ValueError, match="clean Git checkout"):
        package_candidate(
            repo_root=repo,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace=dirty,
            image="example/fugue:test",
            build=False,
        )

    secret = _workspace(tmp_path / "secret", secret=True)
    with pytest.raises(ValueError, match="may contain a secret"):
        package_candidate(
            repo_root=repo,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace=secret,
            image="example/fugue:test",
            build=False,
        )


def test_packaging_rejects_inconsistent_candidate_rows(tmp_path: Path) -> None:
    repo, run_id, candidate_id = _packaging_run(tmp_path)
    workspace = _workspace(tmp_path / "workspace")
    cells_path = repo / ".fugue/runtime" / run_id / "cells.jsonl"
    record = json.loads(cells_path.read_text())
    record["model"] = "openai/different-model"
    cells_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="disagrees on model"):
        package_candidate(
            repo_root=repo,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace=workspace,
            image="example/fugue:test",
            build=False,
        )


def test_context_preparation_uses_only_the_tracked_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, run_id, candidate_id = _packaging_run(tmp_path)
    workspace = _workspace(tmp_path / "workspace")
    original = deployment._prepare_serving_context

    def prepare(**kwargs):
        packaged_workspace = kwargs["workspace"]
        assert (packaged_workspace / "app.py").is_file()
        assert not (packaged_workspace / "ignored.txt").exists()
        return original(**kwargs)

    monkeypatch.setattr(deployment, "_prepare_serving_context", prepare)
    package_candidate(
        repo_root=repo,
        run_id=run_id,
        candidate_id=candidate_id,
        workspace=workspace,
        image="example/fugue:test",
        build=False,
    )


def test_rag_packaging_preserves_candidate_tools_and_uses_local_context() -> None:
    candidate = {
        "context": {"id": "rag-bm25"},
        "agent": {
            "env": {"FUGUE_CONTEXT_ENDPOINT": "http://fugue-context:8000/mcp"},
            "mcp_servers": [
                {"name": "candidate-tool", "url": "https://tools.example/mcp"},
                {"name": "fugue-context", "url": "http://fugue-context:8000/mcp"},
            ],
        },
        "environment": {},
        "prompt_assets": {},
        "skill_assets": {},
    }

    packaged = _deployment_candidate(candidate, prepared_paths=[])

    assert [item["name"] for item in packaged["agent"]["mcp_servers"]] == [
        "candidate-tool",
        "fugue-context",
    ]
    assert packaged["agent"]["mcp_servers"][-1]["transport"] == "stdio"
    assert "FUGUE_CONTEXT_ENDPOINT" not in packaged["agent"]["env"]
    assert packaged["agent"]["env"]["FUGUE_REPO_ROOT"] == "/fugue-src"
