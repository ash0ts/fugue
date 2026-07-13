from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from fugue.bench.context import (
    ContextRuntime,
    ContextSystemSpec,
    RepositorySnapshot,
    RetrievalHit,
    RetrievalQuery,
    _publish_cache_generation,
    context_cache_key,
    get_context_system,
    list_context_systems,
    load_provider,
    preflight_context,
    prepare_context,
)
from fugue.bench.scoring import (
    latency_summary,
    pareto_frontier,
    score_retrieval,
    summarize_metric_rows,
)
from fugue.bench.workloads import (
    RetrievalCase,
    WorkloadDataset,
    load_workload_dataset,
    run_retrieval_workload,
)


def _snapshot() -> RepositorySnapshot:
    return RepositorySnapshot(
        task_id="fixture__task",
        repo="fixture/repo",
        commit="abc123",
        checkout=Path(__file__).parent / "fixtures" / "repo",
    )


def _prepare_in_process(cache_root: str, builder_model: str) -> tuple[str, bool]:
    repo_root = Path(__file__).resolve().parents[1]
    runtime = ContextRuntime(
        repo_root,
        Path(cache_root),
        {"FUGUE_BUILDER_MODEL": builder_model},
    )
    prepared = asyncio.run(
        prepare_context(get_context_system("agentsmd", repo_root), _snapshot(), runtime)
    )
    return prepared.cache_key, prepared.cache_hit


def test_context_library_and_license_gate(tmp_path: Path) -> None:
    systems = {item.id: item for item in list_context_systems(tmp_path)}
    assert {"none", "agentsmd", "rag-bm25", "gitnexus"} <= systems.keys()

    runtime = ContextRuntime(tmp_path, tmp_path / "cache", {})
    checks = asyncio.run(preflight_context(systems["gitnexus"], runtime))
    license_check = next(item for item in checks if item.name == "license")
    assert license_check.ok is False

    approved = ContextRuntime(
        tmp_path,
        tmp_path / "cache",
        {"FUGUE_LICENSE_APPROVED_GITNEXUS": "true"},
    )
    checks = asyncio.run(preflight_context(systems["gitnexus"], approved))
    assert next(item for item in checks if item.name == "license").ok is True


def test_context_cache_is_content_addressed_and_reused(tmp_path: Path) -> None:
    runtime = ContextRuntime(tmp_path, tmp_path / "cache", {})
    spec = get_context_system("agentsmd", tmp_path)
    snapshot = _snapshot()

    first = asyncio.run(prepare_context(spec, snapshot, runtime))
    second = asyncio.run(prepare_context(spec, snapshot, runtime))

    assert first.cache_key == context_cache_key(spec, snapshot)
    assert second.cache_hit is True
    assert (first.path / "artifact" / "AGENTS.md").is_file()
    assert first.metrics["build_latency_ms"] >= 0
    assert first.metrics["cpu_time_sec"] is None or first.metrics["cpu_time_sec"] >= 0
    assert first.metrics["max_memory_mb"] is None or first.metrics["max_memory_mb"] > 0
    assert first.metrics["builder_cost_usd"] is None
    assert first.metrics["index_size_bytes"] > 0
    assert json.loads((first.path / "context-manifest.json").read_text())["snapshot"] == {
        "commit": "abc123",
        "dataset_id": "",
        "repo": "fixture/repo",
        "task_id": "fixture__task",
    }
    assert list(runtime.cache_root.glob(".*.lock"))


def test_failed_context_build_is_not_published(tmp_path: Path) -> None:
    spec = ContextSystemSpec(
        id="failure",
        title="Failure",
        provider="fugue.bench.context:CommandContextProvider",
        version="1",
        capabilities=frozenset({"prepare"}),
        config={
            "prepare": {
                "command": [sys.executable, "-c", "raise RuntimeError('boom')"]
            }
        },
    )
    runtime = ContextRuntime(tmp_path, tmp_path / "cache", {})
    key = context_cache_key(spec, _snapshot())

    with pytest.raises(subprocess.CalledProcessError):
        asyncio.run(prepare_context(spec, _snapshot(), runtime))

    assert not (runtime.cache_root / key).exists()
    leftovers = list(runtime.cache_root.glob(f".{key}.*"))
    assert leftovers == [runtime.cache_root / f".{key}.lock"]


def test_cache_publication_restores_previous_generation_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "cache-key"
    backup = tmp_path / ".cache-key.previous"
    staged = tmp_path / ".cache-key.staged"
    final.mkdir()
    staged.mkdir()
    (final / "value").write_text("stable")
    (staged / "value").write_text("new")
    real_replace = __import__("os").replace

    def fail_new_publication(source, target):
        if Path(source) == staged and Path(target) == final:
            raise OSError("simulated publication crash")
        return real_replace(source, target)

    monkeypatch.setattr("fugue.bench.context.os.replace", fail_new_publication)
    with pytest.raises(OSError, match="simulated"):
        _publish_cache_generation(staged, final, backup)

    assert (final / "value").read_text() == "stable"
    assert not backup.exists()


def test_context_cache_coordinates_same_and_different_keys_across_processes(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    try:
        pool = ProcessPoolExecutor(max_workers=2)
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"multiprocessing locks unavailable: {exc}")
    with pool:
        same = list(
            pool.map(
                _prepare_in_process,
                [cache_root.as_posix(), cache_root.as_posix()],
                ["openai/gpt-5-mini", "openai/gpt-5-mini"],
            )
        )
    assert same[0][0] == same[1][0]
    assert sorted(hit for _, hit in same) == [False, True]

    with ProcessPoolExecutor(max_workers=2) as pool:
        different = list(
            pool.map(
                _prepare_in_process,
                [cache_root.as_posix(), cache_root.as_posix()],
                ["openai/gpt-5", "anthropic/claude-sonnet-4-5"],
            )
        )
    assert different[0][0] != different[1][0]
    index = json.loads((cache_root / "index.json").read_text())
    assert len(index["entries"]) == 3


def test_context_cache_key_tracks_builder_and_embedding_models(tmp_path: Path) -> None:
    spec = get_context_system("rag-bm25", tmp_path)
    snapshot = _snapshot()
    first = ContextRuntime(
        tmp_path,
        tmp_path / "cache",
        {"FUGUE_BUILDER_MODEL": "openai/gpt-5-mini"},
    )
    second = ContextRuntime(
        tmp_path,
        tmp_path / "cache",
        {"FUGUE_BUILDER_MODEL": "anthropic/claude-haiku-4-5"},
    )

    assert context_cache_key(spec, snapshot, first) != context_cache_key(
        spec, snapshot, second
    )


def test_gitnexus_artifact_contains_container_valid_repository_and_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "src").mkdir(parents=True)
    (checkout / "src" / "app.py").write_text("print('ok')\n")
    (checkout / ".git").mkdir()
    output = tmp_path / "output"
    spec = get_context_system("gitnexus", tmp_path)
    runtime = ContextRuntime(
        tmp_path,
        tmp_path / "cache",
        {"FUGUE_LICENSE_APPROVED_GITNEXUS": "true"},
        output_dir=output,
    )

    def fake_run(command, *, cwd, env, check):
        (checkout / ".gitnexus").mkdir()
        (checkout / ".gitnexus" / "graph.json").write_text("{}")
        registry = Path(env["HOME"]) / ".gitnexus" / "registry.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({checkout.as_posix(): {"ready": True}}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("fugue.bench.context.subprocess.run", fake_run)
    provider = load_provider(spec)
    prepared = asyncio.run(
        provider.prepare(
            spec,
            RepositorySnapshot("task", "repo", "commit", checkout, "dataset"),
            runtime,
        )
    )

    repository = prepared.path / "artifact" / "repository"
    assert (repository / "src" / "app.py").is_file()
    assert (repository / ".gitnexus" / "graph.json").is_file()
    registry = prepared.path / "artifact" / "home" / ".gitnexus" / "registry.json"
    assert "/fugue-context/artifact/repository" in registry.read_text()
    assert checkout.as_posix() not in registry.read_text()


def test_retrieval_metrics_and_na_behavior() -> None:
    query = RetrievalQuery(
        id="q1",
        text="Where is answer defined?",
        expected_paths=("src/app.py", "src/other.py"),
    )
    metrics = score_retrieval(
        query,
        [
            RetrievalHit(path="README.md"),
            RetrievalHit(path="src/app.py"),
        ],
    )

    assert metrics["mrr"] == 0.5
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 0.5
    assert score_retrieval(RetrievalQuery("q2", "x"), [])["recall_at_10"] is None
    assert latency_summary([]) == {"p50_ms": None, "p95_ms": None}
    frontier = pareto_frontier(
        [
            {"id": "fast", "quality": 0.8, "cost": 1},
            {"id": "slow", "quality": 0.8, "cost": 2},
            {"id": "best", "quality": 0.9, "cost": 3},
        ],
        quality="quality",
        cost="cost",
    )
    assert {item["id"] for item in frontier} == {"fast", "best"}


def test_workload_dataset_rejects_duplicate_cases_and_invalid_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text(
        """
id: retrieval
runner: retrieval
cases:
  - {id: same, repo: fixture/repo, commit: abc, query: first}
  - {id: same, repo: fixture/repo, commit: abc, query: second}
"""
    )
    with pytest.raises(ValueError, match="duplicate retrieval case"):
        load_workload_dataset(path)

    dataset = WorkloadDataset(
        id="empty",
        runner="retrieval",
        retrieval_cases=(),
    )
    runtime = ContextRuntime(tmp_path, tmp_path / "cache", {})
    with pytest.raises(ValueError, match="attempts must be positive"):
        asyncio.run(
            run_retrieval_workload(
                dataset=dataset,
                system_id="none",
                runtime=runtime,
                experiment_id="experiment",
                preset_id="smoke",
                run_id="run",
                attempts=0,
            )
        )


def test_retrieval_metrics_deduplicate_file_hits_and_stay_bounded() -> None:
    query = RetrievalQuery(
        id="duplicate",
        text="find both files",
        expected_paths=("src/a.py", "src/b.py"),
    )
    metrics = score_retrieval(
        query,
        [
            RetrievalHit(path="src/a.py", start_line=1),
            RetrievalHit(path="./src/a.py", start_line=80),
            RetrievalHit(path="src/b.py", start_line=1),
        ],
    )

    assert metrics["raw_result_count"] == 3
    assert metrics["unique_result_count"] == 2
    for name in ("mrr", "ndcg_at_10", "recall_at_5", "precision_at_5"):
        assert 0 <= float(metrics[name]) <= 1


def test_retrieval_summary_reports_latency_and_failure_rates() -> None:
    summary = summarize_metric_rows(
        [
            {
                "record_type": "retrieval",
                "context_system_id": "rag",
                "query_latency_ms": 10,
                "empty": 0,
            },
            {
                "record_type": "retrieval",
                "context_system_id": "rag",
                "query_latency_ms": 30,
                "empty": 1,
                "exception_class": "TimeoutError",
            },
        ],
        ("context_system_id",),
    )[0]

    assert summary["query_latency_p50_ms"] == 20
    assert summary["query_latency_p95_ms"] == 29
    assert summary["empty_rate"] == 0.5
    assert summary["error_rate"] == 0.5


@pytest.mark.parametrize("system_id", ["none", "markdown-log"])
def test_empty_baselines_emit_scored_retrieval_rows(
    system_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = RepositorySnapshot(
        task_id="q1",
        dataset_id="retrieval-fixture",
        repo="fixture/repo",
        commit="abc123",
        checkout=Path(__file__).parent / "fixtures" / "repo",
    )
    monkeypatch.setattr(
        "fugue.bench.workloads.checkout_repository",
        lambda **kwargs: snapshot,
    )
    dataset = WorkloadDataset(
        id="retrieval-fixture",
        runner="retrieval",
        retrieval_cases=(
            RetrievalCase(
                id="q1",
                repo="fixture/repo",
                commit="abc123",
                query="Where is the implementation?",
                expected_paths=("src/expected.py",),
            ),
        ),
    )
    runtime = ContextRuntime(tmp_path, tmp_path / ".fugue/cache/context/v2", {})

    rows = asyncio.run(
        run_retrieval_workload(
            dataset=dataset,
            system_id=system_id,
            runtime=runtime,
            experiment_id="experiment",
            preset_id="smoke",
            run_id=f"run-{system_id}",
        )
    )

    scored = [row for row in rows if row["record_type"] == "retrieval"]
    assert len(scored) == 1
    assert scored[0]["applicable"] is True
    assert scored[0]["recall_at_1"] == 0.0
