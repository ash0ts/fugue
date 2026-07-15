from pathlib import Path

import pytest

from fugue.bench.manifest import load_manifest


def test_manifest_loads_benchmark_surface_without_experiment_axes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset:
  ref: swe-bench/swe-bench-verified
  version: latest
harnesses:
  - name: hermes
    agent: fugue.agents:FugueHermes
tasks:
  - id: astropy__astropy-12907
    repo: astropy/astropy
    base_commit: d16bfe05a744909de4b27f5875fe0d4ed41ce607
"""
    )

    manifest = load_manifest(manifest_path)

    assert manifest.dataset.harbor_ref == "swe-bench/swe-bench-verified@latest"
    assert manifest.select_harnesses(["hermes"])[0].agent.endswith("FugueHermes")
    assert manifest.tasks[0].repo_slug == "astropy/astropy"


def test_manifest_supports_materialized_local_datasets(tmp_path: Path) -> None:
    manifest_path = tmp_path / "qa.yaml"
    manifest_path.write_text(
        """
dataset:
  path: .fugue/cache/datasets/qa/v1
  materializer: package.module:Adapter
  source: {url: https://example.test/data.jsonl, sha256: abc}
harnesses:
  - name: codex
    agent: fugue.agents:FugueCodex
tasks:
  - id: qa-001
    repo: org/repo
    base_commit: abc123
    metadata: {source_index: 7}
"""
    )

    manifest = load_manifest(manifest_path)

    assert manifest.dataset.ref is None
    assert manifest.dataset.path == Path(".fugue/cache/datasets/qa/v1")
    assert manifest.dataset.materializer == "package.module:Adapter"
    assert manifest.tasks[0].metadata == {"source_index": 7}


def test_manifest_normalizes_typed_repository_and_http_source(tmp_path: Path) -> None:
    manifest_path = tmp_path / "typed.yaml"
    commit = "a" * 40
    digest = "b" * 64
    manifest_path.write_text(
        f"""
dataset:
  path: .fugue/cache/datasets/typed
  source:
    type: http
    url: https://example.test/tasks.jsonl
    sha256: {digest}
    license: MIT
harnesses:
  - {{name: codex, agent: fugue.agents:FugueCodex}}
tasks:
  - id: typed-task
    repository:
      type: git
      url: https://github.com/example/project.git
      commit: {commit}
      path: packages/core
"""
    )

    manifest = load_manifest(manifest_path)

    assert manifest.dataset.source_spec is not None
    assert manifest.dataset.source["type"] == "http"
    task = manifest.tasks[0]
    assert task.repo == "example/project"
    assert task.base_commit == commit
    assert task.repository is not None
    assert task.repository.to_dict() == {
        "type": "git",
        "url": "https://github.com/example/project",
        "commit": commit,
        "path": "packages/core",
    }


def test_typed_repository_rejects_moving_refs(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
dataset: {ref: test/tasks}
harnesses: [{name: codex, agent: fugue.agents:FugueCodex}]
tasks:
  - id: task
    repository:
      type: git
      url: https://github.com/example/project
      commit: main
"""
    )
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        load_manifest(path)


def test_pilot_canary_declares_gold_evidence_paths() -> None:
    manifest = load_manifest(Path(__file__).parents[1] / "datasets" / "pilot.yaml")
    task = next(
        item for item in manifest.tasks if item.id == "astropy__astropy-12907"
    )

    assert task.expected_paths == (
        "astropy/modeling/separable.py",
        "astropy/modeling/tests/test_separable.py",
    )


def test_skillsbench_artifacts_are_scoped_to_the_task_that_produces_them() -> None:
    manifest = load_manifest(
        Path(__file__).parents[1] / "datasets" / "skillsbench-pdf.yaml"
    )

    assert {task.id: task.artifacts for task in manifest.tasks} == {
        "court-form-filling": ("/root/sc100-filled.pdf",),
        "pdf-excel-diff": ("/root/diff_report.json",),
        "paper-anonymizer": ("/root/redacted",),
    }


def test_manifest_rejects_scalar_task_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "bad-artifacts.yaml"
    path.write_text(
        """
dataset: {ref: test/dataset}
harnesses: [{name: codex, agent: fugue.agents:FugueCodex}]
tasks: [{id: task, artifacts: /root/result.json}]
"""
    )

    with pytest.raises(ValueError, match="task artifacts must be a list"):
        load_manifest(path)


def test_manifest_rejects_duplicate_ids_and_nonpositive_execution_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
dataset: {ref: test/dataset}
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks: [{id: task}]
k: 0
"""
    )
    with pytest.raises(ValueError, match="duplicate harness"):
        load_manifest(path)

    path.write_text(
        """
dataset: {ref: test/dataset}
harnesses: [{name: codex, agent: fugue.agents:FugueCodex}]
tasks: [{id: task}]
k: 0
"""
    )
    with pytest.raises(ValueError, match="k must be positive"):
        load_manifest(path)
