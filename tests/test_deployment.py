from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_operator import make_operator_repo

from fugue.bench import deployment
from fugue.bench.deployment import (
    _deployment_candidate,
    _write_assets,
    candidate_packageability,
    package_candidate,
)
from fugue.bench.execution import plan_cells, write_run_manifest
from fugue.bench.operator import ExperimentRequest
from fugue.bench.reproducibility import build_run_snapshot, write_run_input_lock


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
    shutil.copyfile(Path(__file__).parents[1] / "LICENSE", repo / "LICENSE")
    shutil.copytree(Path(__file__).parents[1] / "fugue", repo / "fugue")
    (repo / ".gitignore").write_text(".fugue/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Fugue Tests",
        "-c",
        "user.email=fugue@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    experiment = service.experiment("demo")
    variant = replace(
        experiment.variants[0],
        prompt_id="demo-prompt",
        skills=["demo-skill"],
        agent_env={
            "CUSTOM_TOKEN": "trace-secret-value",
            "SHORT_SECRET": "x",
        },
    )
    experiment = replace(experiment, variants=[variant], n_attempts=2)
    run_id = "run-package"
    request = ExperimentRequest(experiment_id="demo", n_attempts=2)
    jobs = service.rendered_jobs(
        request,
        run_id=run_id,
        experiment=experiment,
    )
    cells = plan_cells(jobs, run_id=run_id, run_name="package fixture")
    snapshot = build_run_snapshot(
        repo_root=repo,
        run_id=run_id,
        experiment=experiment,
        request=asdict(request),
        jobs=jobs,
        cells=cells,
        env={"CUSTOM_TOKEN": "trace-secret-value", "SHORT_SECRET": "x"},
    )
    write_run_input_lock(repo, snapshot)
    write_run_manifest(
        repo,
        run_id,
        {
            "status": run_status,
            "run_name": "package fixture",
            "experiment_id": "demo",
        },
    )
    cells_path = repo / ".fugue/runtime" / run_id / "cells.jsonl"
    cells_path.write_text(
        "\n".join(
            json.dumps(
                cell.record(
                    "passed" if index == 0 else "failed",
                    benchmark_outcome="passed" if index == 0 else "unscored",
                    reward=1.0 if index == 0 else None,
                )
            )
            for index, cell in enumerate(cells)
        )
        + "\n"
    )
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
        allow_failed=True,
        build=False,
    )
    second = package_candidate(
        repo_root=repo,
        run_id=run_id,
        candidate_id=candidate_id,
        workspace=workspace,
        image="example/fugue:test",
        allow_failed=True,
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
    assert "raw.githubusercontent.com" in spec["network_allowed_hosts"]
    assert "nodejs.org" in spec["network_allowed_hosts"]
    assert spec["provenance"]["workspace"]["digest"] == first.workspace_digest
    assert spec["candidate"]["model_route"]["responses_base_url"] == (
        "https://api.openai.com/v1"
    )
    dockerfile = (first.path / "Dockerfile").read_text()
    assert "io.fugue.candidate.id" in dockerfile
    assert "io.fugue.input-lock.digest" in dockerfile
    assert "io.fugue.runtime.digest" in dockerfile
    assert "python:3.13-slim" in dockerfile
    assert "docker-buildx docker-cli docker-compose" in dockerfile
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
            allow_failed=True,
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
            allow_failed=True,
            build=False,
        )


def test_packaging_rejects_input_lock_tampering(tmp_path: Path) -> None:
    repo, run_id, candidate_id = _packaging_run(tmp_path)
    workspace = _workspace(tmp_path / "workspace")
    lock_path = repo / ".fugue/runtime" / run_id / "input-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["runtime"]["injected"] = True
    lock_path.write_text(json.dumps(lock))

    with pytest.raises(ValueError, match="input lock is corrupt"):
        package_candidate(
            repo_root=repo,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace=workspace,
            image="example/fugue:test",
            allow_failed=True,
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
            allow_failed=True,
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
            allow_failed=True,
            build=False,
        )


def test_packaging_rejects_dirty_runtime_source(tmp_path: Path) -> None:
    repo, run_id, candidate_id = _packaging_run(tmp_path)
    workspace = _workspace(tmp_path / "workspace")
    (repo / "fugue/__init__.py").write_text("# dirty runtime\n")

    with pytest.raises(ValueError, match="runtime source must be a clean"):
        package_candidate(
            repo_root=repo,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace=workspace,
            image="example/fugue:test",
            allow_failed=True,
            build=False,
        )


def test_packaging_rejects_escaping_symlink_and_credential_remote(
    tmp_path: Path,
) -> None:
    repo, run_id, candidate_id = _packaging_run(tmp_path)
    workspace = _workspace(tmp_path / "workspace")
    (workspace / "app.py").unlink()
    (workspace / "app.py").symlink_to("../outside.py")
    _git(workspace, "add", "app.py")
    _git(
        workspace,
        "-c",
        "user.name=Fugue Tests",
        "-c",
        "user.email=fugue@example.com",
        "commit",
        "-qm",
        "escaping link",
    )

    with pytest.raises(ValueError, match="symlink escapes"):
        package_candidate(
            repo_root=repo,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace=workspace,
            image="example/fugue:test",
            allow_failed=True,
            build=False,
        )

    safe = _workspace(tmp_path / "remote")
    _git(safe, "remote", "add", "origin", "https://token:secret@example.com/repo.git")
    with pytest.raises(ValueError, match="credential-bearing Git remote"):
        package_candidate(
            repo_root=repo,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace=safe,
            image="example/fugue:test",
            allow_failed=True,
            build=False,
        )


@pytest.mark.parametrize(
    ("planned", "records", "allow_failed", "expected", "reason"),
    (
        ([], [], False, False, "not present"),
        ([{"cell_id": "n", "applicable": False}], [], False, False, "not applicable"),
        ([{"cell_id": "a"}], [], False, False, "no durable state"),
        ([{"cell_id": "a"}], [{"cell_id": "a", "status": "running"}], False, False, "not terminal"),
        ([{"cell_id": "a"}], [{"cell_id": "a", "status": "failed"}], True, False, "no passed"),
        (
            [{"cell_id": "a"}, {"cell_id": "b"}],
            [
                {
                    "cell_id": "a",
                    "status": "passed",
                    "benchmark_outcome": "passed",
                },
                {"cell_id": "b", "status": "failed"},
            ],
            False,
            False,
            "--allow-failed",
        ),
        (
            [{"cell_id": "a"}, {"cell_id": "b"}],
            [
                {
                    "cell_id": "a",
                    "status": "passed",
                    "benchmark_outcome": "passed",
                },
                {"cell_id": "b", "status": "failed"},
            ],
            True,
            True,
            "explicitly allowed",
        ),
    ),
)
def test_candidate_packageability_explains_every_terminal_state(
    planned: list[dict],
    records: list[dict],
    allow_failed: bool,
    expected: bool,
    reason: str,
) -> None:
    snapshot = {
        "planned_matrix": [
            {**item, "candidate_id": "candidate-a"} for item in planned
        ]
    }

    packageable, detail = candidate_packageability(
        snapshot,
        records,
        "candidate-a",
        allow_failed=allow_failed,
    )

    assert packageable is expected
    assert reason in detail


def test_candidate_packageability_uses_deterministic_outcome() -> None:
    snapshot = {
        "planned_matrix": [
            {"cell_id": "pass", "candidate_id": "candidate-a"},
            {"cell_id": "fail", "candidate_id": "candidate-a"},
        ]
    }
    records = [
        {
            "cell_id": "pass",
            "status": "passed",
            "benchmark_outcome": "passed",
        },
        {
            "cell_id": "fail",
            "status": "passed",
            "benchmark_outcome": "failed",
        },
    ]

    blocked, reason = candidate_packageability(snapshot, records, "candidate-a")
    allowed, allowed_reason = candidate_packageability(
        snapshot, records, "candidate-a", allow_failed=True
    )

    assert blocked is False
    assert "1 failed benchmark cell(s)" in reason
    assert allowed is True
    assert "explicitly allowed" in allowed_reason


def test_candidate_packageability_requires_a_pass_but_allows_terminal_unscored() -> None:
    snapshot = {
        "candidate_runtime": {"candidate-a": {"harness": "codex"}},
        "planned_matrix": [
            {"cell_id": "a", "candidate_id": "candidate-a"},
            {"cell_id": "b", "candidate_id": "candidate-a"},
        ],
    }
    records = [
        {"cell_id": "a", "status": "passed", "benchmark_outcome": "passed"},
        {"cell_id": "b", "status": "passed", "benchmark_outcome": "unscored"},
    ]

    packageable, reason = candidate_packageability(
        snapshot, records, "candidate-a", allow_failed=True
    )
    assert packageable is True
    assert reason == (
        "packageable with 1 passed and 1 unscored terminal applicable cell(s)"
    )

    only_unscored = candidate_packageability(
        {
            "candidate_runtime": {"candidate-a": {"harness": "codex"}},
            "planned_matrix": [
                {"cell_id": "b", "candidate_id": "candidate-a"}
            ],
        },
        [records[1]],
        "candidate-a",
    )
    assert only_unscored == (False, "candidate has no passed applicable cells")

    snapshot["candidate_runtime"]["candidate-a"]["harness"] = "direct"
    packageable, reason = candidate_packageability(
        snapshot, records, "candidate-a", allow_failed=True
    )
    assert packageable is False
    assert reason == "candidate harness is not supported for serving: direct"


def test_packaging_rejects_inconsistent_candidate_rows(tmp_path: Path) -> None:
    repo, run_id, candidate_id = _packaging_run(tmp_path)
    workspace = _workspace(tmp_path / "workspace")
    cells_path = repo / ".fugue/runtime" / run_id / "cells.jsonl"
    records = [json.loads(line) for line in cells_path.read_text().splitlines()]
    records[0]["model"] = "openai/different-model"
    cells_path.write_text("\n".join(json.dumps(row) for row in records) + "\n")

    with pytest.raises(ValueError, match="disagrees on model"):
        package_candidate(
            repo_root=repo,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace=workspace,
            image="example/fugue:test",
            allow_failed=True,
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
        allow_failed=True,
        build=False,
    )


def test_packaging_materializes_an_empty_asset_root(tmp_path: Path) -> None:
    destination = tmp_path / "assets"

    _write_assets({"prompt_assets": {}, "skill_assets": {}}, destination)

    assert destination.is_dir()


def test_packaging_rejects_mcp_without_a_declared_serving_contract() -> None:
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

    with pytest.raises(ValueError, match="MCP-free serving delivery"):
        _deployment_candidate(candidate, prepared_paths=[])
