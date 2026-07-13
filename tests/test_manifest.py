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
