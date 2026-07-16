from pathlib import Path

import pytest

from fugue.bench.manifest import fixture_repository_digest, load_manifest


def test_manifest_loads_benchmark_surface_without_experiment_axes(
    tmp_path: Path,
) -> None:
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
    repository: {type: git, url: https://github.com/astropy/astropy, commit: d16bfe05a744909de4b27f5875fe0d4ed41ce607}
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
    repository: {type: git, url: https://github.com/org/repo, commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}
    metadata: {source_index: 7}
"""
    )

    manifest = load_manifest(manifest_path)

    assert manifest.dataset.ref is None
    assert manifest.dataset.path == Path(".fugue/cache/datasets/qa/v1")
    assert manifest.dataset.materializer == "package.module:Adapter"
    assert manifest.tasks[0].metadata == {"source_index": 7}


def test_manifest_loads_pinned_dataset_verifier_runtime(tmp_path: Path) -> None:
    path = tmp_path / "offline.yaml"
    path.write_text(
        """
dataset:
  ref: swe-bench/swe-bench-verified
  version: sha256:abc
  verifier_runtime:
    profile: swebench-v4-offline
    python_interpreter: /opt/fugue-verifier/bin/python
    python_packages: [swebench==4.0.3, datasets==2.16.1, fastcore==1.10.5]
harnesses: [{name: codex, agent: fugue.agents:FugueCodex}]
tasks: [{id: pydata__xarray-6992}]
"""
    )

    runtime = load_manifest(path).dataset.verifier_runtime

    assert runtime is not None
    assert runtime.to_dict() == {
        "profile": "swebench-v4-offline",
        "python_interpreter": "/opt/fugue-verifier/bin/python",
        "python_packages": [
            "swebench==4.0.3",
            "datasets==2.16.1",
            "fastcore==1.10.5",
        ],
    }


def test_manifest_rejects_unpinned_dataset_verifier_package(tmp_path: Path) -> None:
    path = tmp_path / "offline.yaml"
    path.write_text(
        """
dataset:
  ref: swe-bench/swe-bench-verified
  verifier_runtime:
    profile: swebench-v4-offline
    python_interpreter: /opt/fugue-verifier/bin/python
    python_packages: [swebench>=4]
harnesses: [{name: codex, agent: fugue.agents:FugueCodex}]
tasks: [{id: task}]
"""
    )

    with pytest.raises(ValueError, match="exact name==version pins"):
        load_manifest(path)


@pytest.mark.parametrize(
    "name",
    [
        "swe-bench-hard-discovery-v2.yaml",
        "swe-bench-hard-holdout-v2.yaml",
        "swe-bench-gitnexus-ablation-v2.yaml",
        "swe-bench-qualification-v1.yaml",
        "swe-bench-controls-v1.yaml",
    ],
)
def test_hard_swe_manifests_lock_the_offline_verifier(name: str) -> None:
    path = Path(__file__).parents[1] / "datasets" / "repo-memory" / name

    runtime = load_manifest(path).dataset.verifier_runtime

    assert runtime is not None
    assert runtime.profile == "swebench-v4-offline"
    assert runtime.python_interpreter == "/opt/fugue-verifier/bin/python"
    assert runtime.python_packages == (
        "swebench==4.0.3",
        "datasets==2.16.1",
        "fastcore==1.10.5",
    )


def test_manifest_rejects_mixed_dataset_and_task_verifier_runtimes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed.yaml"
    path.write_text(
        f"""
dataset:
  ref: swe-bench/swe-bench-verified
  verifier_runtime:
    profile: swebench-v4-offline
    python_interpreter: /opt/fugue-verifier/bin/python
    python_packages: [swebench==4.0.3, datasets==2.16.1, fastcore==1.10.5]
harnesses: [{{name: codex, agent: fugue.agents:FugueCodex}}]
tasks:
  - id: task
    verifier_runtime:
      python_packages: [pytest==8.4.1]
      test_script_sha256: {"a" * 64}
"""
    )

    with pytest.raises(ValueError, match="may not be mixed"):
        load_manifest(path)


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


def test_manifest_rejects_legacy_task_repository_fields(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """
dataset: {ref: test/tasks}
harnesses: [{name: codex, agent: fugue.agents:FugueCodex}]
tasks:
  - id: task
    repo: example/project
    base_commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
"""
    )

    with pytest.raises(ValueError, match="must use repository"):
        load_manifest(path)


def test_manifest_supports_content_addressed_fixture_repository(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "module.py").write_text("VALUE = 1\n")
    digest = fixture_repository_digest(fixture)
    path = tmp_path / "fixture.yaml"
    path.write_text(
        f"""
dataset: {{path: local-tasks}}
harnesses: [{{name: codex, agent: fugue.agents:FugueCodex}}]
tasks:
  - id: vector-contract
    repository:
      type: fixture
      path: fixture
      sha256: {digest}
"""
    )

    task = load_manifest(path).tasks[0]

    assert task.repo == "fixture/fixture"
    assert task.base_commit == digest
    assert task.repository is not None
    assert task.repository.to_dict() == {
        "type": "fixture",
        "path": "fixture",
        "sha256": digest,
    }


def test_pilot_canary_declares_gold_evidence_paths() -> None:
    manifest = load_manifest(Path(__file__).parents[1] / "datasets" / "pilot.yaml")
    task = next(item for item in manifest.tasks if item.id == "astropy__astropy-12907")

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
