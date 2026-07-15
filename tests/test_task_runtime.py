from __future__ import annotations

import json
import subprocess
from pathlib import Path

import toml

from fugue.bench import task_runtime
from fugue.bench.manifest import load_manifest


def _fixture(tmp_path: Path):
    dataset = tmp_path / "dataset" / "task-one"
    (dataset / "environment").mkdir(parents=True)
    (dataset / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (dataset / "instruction.md").write_text("Do the task.\n")
    (dataset / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "task-one"\n\n'
        "[environment]\nallow_internet = true\n"
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "dataset:\n  path: dataset\n"
        "harnesses:\n  - {name: codex, agent: fugue.agents:FugueCodex}\n"
        "tasks:\n  - id: task-one\n    metadata: {architecture: arm64}\n"
    )
    return load_manifest(manifest_path), dataset


def test_task_preparation_locks_image_and_disables_trial_network(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, source = _fixture(tmp_path)
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    [
                        {
                            "Id": "sha256:" + "a" * 64,
                            "Architecture": "arm64",
                            "Os": "linux",
                        }
                    ]
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(task_runtime.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(task_runtime, "_resolve_task_source", lambda *args: source)
    monkeypatch.setattr(task_runtime.subprocess, "run", run)
    lock = task_runtime.prepare_task_runtime(
        manifest, manifest.tasks[0], repo_root=tmp_path
    )

    prepared = Path(lock["dataset_path"]) / "task-one" / "task.toml"
    config = toml.loads(prepared.read_text())
    assert config["environment"]["docker_image"] == "sha256:" + "a" * 64
    assert config["environment"]["network_mode"] == "no-network"
    assert config["environment"]["allowed_hosts"] == []
    assert "allow_internet" not in config["environment"]
    assert commands[0][0:5] == [
        "docker",
        "build",
        "--provenance=false",
        "--platform",
        "linux/arm64",
    ]


def test_task_runtime_lock_rejects_dataset_drift(tmp_path: Path) -> None:
    manifest, _source = _fixture(tmp_path)
    root = task_runtime._lock_root(manifest, manifest.tasks[0], tmp_path)
    dataset = root / "dataset-deadbeef"
    dataset.mkdir(parents=True)
    path = root / "runtime-lock-arm64.json"
    value = {
        "schema_version": 1,
        "contract_version": task_runtime.TASK_RUNTIME_CONTRACT_VERSION,
        "dataset": manifest.dataset.harbor_ref,
        "task_id": "task-one",
        "architecture": "arm64",
        "image_id": "sha256:" + "a" * 64,
        "dataset_path": dataset.as_posix(),
    }
    path.write_text(json.dumps(value))
    assert task_runtime.read_task_runtime_lock(manifest, manifest.tasks[0], tmp_path)

    value["dataset"] = "different"
    path.write_text(json.dumps(value))
    assert (
        task_runtime.read_task_runtime_lock(manifest, manifest.tasks[0], tmp_path)
        is None
    )


def test_task_runtime_lock_rejects_prior_execution_policy(tmp_path: Path) -> None:
    manifest, _source = _fixture(tmp_path)
    root = task_runtime._lock_root(manifest, manifest.tasks[0], tmp_path)
    dataset = root / "dataset-deadbeef"
    dataset.mkdir(parents=True)
    value = {
        "schema_version": 1,
        "contract_version": task_runtime.TASK_RUNTIME_CONTRACT_VERSION - 1,
        "dataset": manifest.dataset.harbor_ref,
        "task_id": "task-one",
        "architecture": "arm64",
        "image_id": "sha256:" + "a" * 64,
        "dataset_path": dataset.as_posix(),
    }
    (root / "runtime-lock-arm64.json").write_text(json.dumps(value))

    assert task_runtime.read_task_runtime_lock(manifest, manifest.tasks[0], tmp_path) is None


def test_local_task_source_does_not_import_harbor(tmp_path: Path) -> None:
    manifest, source = _fixture(tmp_path)

    assert task_runtime._resolve_task_source(
        manifest, manifest.tasks[0], tmp_path
    ) == source.resolve()
