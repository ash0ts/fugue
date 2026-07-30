from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from fugue.bench import datasets
from fugue.bench.datasets import DATASET_MANIFEST, materialize_manifest_dataset
from fugue.bench.manifest import load_manifest


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "qa.yaml"
    path.write_text(
        """
dataset:
  path: .fugue/cache/datasets/qa/revision
  materializer: fugue.bench.datasets:SweQaProMaterializer
  source:
    url: https://example.test/data.jsonl
    sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
harnesses:
  - name: codex
    agent: fugue.agents:FugueCodex
tasks:
  - id: swe-qa-pro-000-fixture
    repository: {type: git, url: https://github.com/fixture/repo, commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}
    metadata: {source_index: 0}
"""
    )
    return path


def test_materialized_harbor_dataset_is_atomic_and_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {
        "repo": "fixture/repo",
        "commit_id": "a" * 40,
        "question": "Where is the fixture implemented?",
        "answer": "The fixture is in src/fixture.py.",
    }

    def write_source(source: dict, destination: Path) -> None:
        destination.write_text(json.dumps(row) + "\n")

    monkeypatch.setattr(datasets, "_download_source", write_source)
    manifest = load_manifest(_manifest(tmp_path))

    first = materialize_manifest_dataset(manifest, tmp_path)
    second = materialize_manifest_dataset(manifest, tmp_path)
    marker = json.loads((first / DATASET_MANIFEST).read_text())
    marker["fingerprint"] = datasets._legacy_dataset_fingerprint(
        manifest, manifest.dataset.source
    )
    (first / DATASET_MANIFEST).write_text(json.dumps(marker))
    mirrored = replace(
        manifest,
        dataset=replace(
            manifest.dataset,
            source={
                "url": "https://mirror.example.test/data.jsonl",
                "sha256": "a" * 64,
            },
        ),
    )
    third = materialize_manifest_dataset(mirrored, tmp_path)

    assert first == second == third
    assert first is not None
    task = first / "swe-qa-pro-000-fixture"
    assert (task / "task.toml").is_file()
    assert "Where is the fixture" in (task / "instruction.md").read_text()
    assert "src/fixture.py" not in (task / "instruction.md").read_text()
    assert "src/fixture.py" in (task / "solution" / "reference.md").read_text()
    assert json.loads((first / DATASET_MANIFEST).read_text())["metrics"] == {
        "source_rows": 1,
        "tasks": 1,
    }
    assert not list(first.parent.glob(".*"))


def test_materializer_rejects_source_drift_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {
        "repo": "different/repo",
        "commit_id": "b" * 40,
        "question": "Question",
        "answer": "Answer",
    }

    def write_source(source: dict, destination: Path) -> None:
        destination.write_text(json.dumps(row) + "\n")

    monkeypatch.setattr(datasets, "_download_source", write_source)
    manifest = load_manifest(_manifest(tmp_path))
    destination = tmp_path / manifest.dataset.path

    with pytest.raises(ValueError, match="source row repo/commit changed"):
        materialize_manifest_dataset(manifest, tmp_path)

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".lock").exists()


def test_cached_authored_dataset_reuses_only_byte_equivalent_materialization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "public-cases.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "task-one",
                "title": "Task one",
                "instruction": "Return one bounded answer.",
                "attachments": [],
                "environment": {
                    "base_image": "python:3.12-slim",
                    "kind": "artifact",
                    "cpus": 1,
                    "memory_mb": 512,
                    "storage_mb": 1024,
                },
                "interaction": {"timeout_sec": 60},
            },
            sort_keys=True,
        )
        + "\n"
    )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    def manifest(criteria_digest: str, profile_digest: str):
        return load_manifest(
            tmp_path / "authored.yaml",
            text=f"""
dataset:
  path: .fugue/cache/authored/{source_sha}
  materializer: fugue.bench.task_authoring:AuthoredTaskMaterializer
  source:
    path: public-cases.jsonl
    sha256: {source_sha}
harnesses:
  - name: claude-code
    agent: fugue.agents:FugueClaudeCode
tasks:
  - id: task-one
    metadata:
      source_index: 0
      task_authoring:
        task_definition_digest: {source_sha}
        criteria_digest: {criteria_digest}
        profile_digests:
          comparison-evaluator: {profile_digest}
""",
        )

    first_manifest = manifest("a" * 64, "b" * 64)
    destination = materialize_manifest_dataset(first_manifest, tmp_path)
    assert destination is not None
    instruction = destination / "task-one/instruction.md"
    original = instruction.read_bytes()

    second_manifest = manifest("c" * 64, "d" * 64)
    assert materialize_manifest_dataset(second_manifest, tmp_path) == destination
    assert instruction.read_bytes() == original
    marker = json.loads((destination / DATASET_MANIFEST).read_text())
    assert marker["fingerprint"] == datasets._dataset_fingerprint(second_manifest)

    instruction.write_text("tampered\n")
    third_manifest = manifest("e" * 64, "f" * 64)
    with pytest.raises(ValueError, match="does not match its manifest"):
        materialize_manifest_dataset(third_manifest, tmp_path)


def test_materialized_dataset_rejects_symlinked_root_and_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "repo": "fixture/repo",
        "commit_id": "a" * 40,
        "question": "Where is the fixture implemented?",
        "answer": "The fixture is in src/fixture.py.",
    }

    def write_source(source: dict, destination: Path) -> None:
        destination.write_text(json.dumps(row) + "\n")

    monkeypatch.setattr(datasets, "_download_source", write_source)
    manifest = load_manifest(_manifest(tmp_path))
    destination = tmp_path / manifest.dataset.path
    external = tmp_path / "external-dataset"
    external.mkdir()
    destination.parent.mkdir(parents=True)
    destination.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="root may not be a symlink"):
        materialize_manifest_dataset(manifest, tmp_path)

    destination.unlink()
    materialized = materialize_manifest_dataset(manifest, tmp_path)
    assert materialized is not None
    marker = materialized / DATASET_MANIFEST
    marker_contents = marker.read_text()
    marker.unlink()
    external_marker = tmp_path / "external-marker.json"
    external_marker.write_text(marker_contents)
    marker.symlink_to(external_marker)

    with pytest.raises(
        ValueError,
        match="root and marker must not be symlinks",
    ):
        materialize_manifest_dataset(manifest, tmp_path)


def test_materialized_tree_digest_rejects_special_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    fifo = root / "private-stream"
    os.mkfifo(fifo)
    try:
        with pytest.raises(
            ValueError,
            match="only directories and regular files",
        ):
            datasets._materialized_tree_digest(root)
    finally:
        fifo.unlink()


def test_materialized_tree_digest_rejects_nested_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    nested = root / "task"
    nested.mkdir(parents=True)
    external = tmp_path / "private-reference.txt"
    external.write_text("host-only truth\n")
    (nested / "reference.md").symlink_to(external)

    with pytest.raises(
        ValueError,
        match="may not contain symlinks",
    ):
        datasets._materialized_tree_digest(root)


def test_gitnexus_contract_verifier_scores_every_terminal_answer(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).parents[1]
        / "datasets/repo-memory/gitnexus-contract"
        / "gitnexus-vector-lexical-mismatch/tests/test.sh"
    )
    log_root = tmp_path / "logs"
    script = tmp_path / "test.sh"
    script.write_text(source.read_text().replace("/logs", str(log_root)))

    def verify(answer: str | None) -> dict[str, float]:
        artifact = log_root / "artifacts/fugue-answer.txt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if answer is None:
            artifact.unlink(missing_ok=True)
        else:
            artifact.write_text(answer)
        subprocess.run(["sh", str(script)], check=True)
        return json.loads((log_root / "verifier/reward.json").read_text())

    assert verify(None) == {"reward": 0.0, "path_resolution": 0.0}
    assert verify("src/relay/blue_quartz.py\n") == {
        "reward": 0.0,
        "path_resolution": 0.0,
    }
    assert verify("src/relay/amber_lantern.py\n") == {
        "reward": 1.0,
        "path_resolution": 1.0,
    }
