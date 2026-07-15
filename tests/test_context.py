from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue import context_client
from fugue.bench.context import (
    AiderRepoMapContextProvider,
    ContextRuntime,
    ContextSystemSpec,
    PreparedContext,
    RepositorySnapshot,
    RetrievalHit,
    RetrievalQuery,
    TrialContext,
    _command_env,
    _copy_repository_snapshot,
    _dense_artifact_contract,
    _materialize_dense_artifact,
    _mem0_config,
    _Mem0FastEmbedder,
    _publish_cache_generation,
    _repository_chunks,
    _resolved_embedding_model,
    _run_context_command,
    bind_context,
    context_cache_key,
    get_context_system,
    list_context_systems,
    load_provider,
    preflight_context,
    prepare_context,
)
from fugue.bench.export import export_rows
from fugue.bench.scoring import (
    latency_summary,
    score_retrieval,
)
from fugue.bench.workloads import (
    RetrievalCase,
    WorkloadDataset,
    _write_rows,
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


def _commit_repository(root: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fugue Test",
            "-c",
            "user.email=fugue@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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


def test_mem0_declares_and_uses_the_shared_fastembed_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = get_context_system("mem0", repo_root)
    assert {"mem0", "qdrant_client", "fastembed"} <= set(spec.required_packages)
    assert spec.config["embedding_provider"] == "fastembed"
    config = _mem0_config(
        spec,
        tmp_path / "memory",
        ContextRuntime(
            repo_root,
            tmp_path / "cache",
            {
                "FUGUE_BRIDGE_BASE_URL": "http://127.0.0.1:4000",
                "FUGUE_BRIDGE_MASTER_KEY": "test-key",
            },
        ),
    )
    assert config["embedder"] == {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-small-en-v1.5",
            "embedding_dims": 384,
        },
    }

    calls: list[tuple[str, tuple[str, ...], int]] = []

    class FakeVector(list[float]):
        def tolist(self) -> list[float]:
            return list(self)

    class FakeTextEmbedding:
        def __init__(self, *, model_name: str, providers: list[str]) -> None:
            calls.append((model_name, tuple(providers), 0))

        def embed(self, values: list[str], *, batch_size: int):
            calls.append((values[0], (), batch_size))
            return iter([FakeVector([0.25, 0.75])])

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )
    embedder = _Mem0FastEmbedder(
        SimpleNamespace(model="BAAI/bge-small-en-v1.5", embedding_dims=2)
    )
    assert embedder.embed("remember this", "add") == [0.25, 0.75]
    assert calls == [
        ("BAAI/bge-small-en-v1.5", ("CPUExecutionProvider",), 0),
        ("remember this", (), 1),
    ]


def test_graphiti_binding_passes_required_env_by_reference(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = get_context_system("graphiti", repo_root)
    assert spec.support == "experimental"
    prepared = PreparedContext("graphiti", "cache", tmp_path, {}, {})
    runtime = ContextRuntime(
        repo_root,
        tmp_path / "cache",
        {
            "FUGUE_GRAPHITI_URI": "bolt://127.0.0.1:7687",
            "FUGUE_GRAPHITI_USER": "neo4j",
            "FUGUE_GRAPHITI_PASSWORD": "private-password",
        },
    )

    binding = asyncio.run(
        bind_context(
            spec,
            prepared,
            TrialContext("experiment", "workload", "task", "harness"),
            runtime,
            delivery="native_mcp",
        )
    )

    assert binding.env == {name: f"${{{name}}}" for name in spec.required_env}
    assert "private-password" not in json.dumps(binding.env)


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
    assert json.loads((first.path / "context-manifest.json").read_text())[
        "snapshot"
    ] == {
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
        deliveries=frozenset({"portable"}),
        config={
            "prepare": {"command": [sys.executable, "-c", "raise RuntimeError('boom')"]}
        },
    )
    runtime = ContextRuntime(tmp_path, tmp_path / "cache", {})
    key = context_cache_key(spec, _snapshot())

    with pytest.raises(subprocess.CalledProcessError):
        asyncio.run(prepare_context(spec, _snapshot(), runtime))

    assert not (runtime.cache_root / key).exists()
    leftovers = list(runtime.cache_root.glob(f".{key}.*"))
    assert leftovers == [runtime.cache_root / f".{key}.lock"]


def test_command_context_resolves_the_bridge_default_credential() -> None:
    env = _command_env(
        {},
        {
            "OPENAI_COMPATIBLE_API_KEY": "${LITELLM_MASTER_KEY}",
            "UNSET_EXTERNAL_KEY": "${EXTERNAL_KEY}",
        },
    )

    assert env["OPENAI_COMPATIBLE_API_KEY"] == "sk-fugue-local"
    assert env["UNSET_EXTERNAL_KEY"] == ""


def test_aider_repomap_is_pinned_and_non_interactive() -> None:
    spec = get_context_system("aider-repomap")
    prepare = spec.config["prepare"]
    command = prepare["command"]

    assert command[:4] == ["uvx", "--from", "aider-chat==0.86.2", "aider"]
    assert command[command.index("--model") + 1] == "openai/fugue-builder"
    assert command[command.index("--openai-api-base") + 1] == (
        "http://127.0.0.1:4000/v1"
    )
    assert command[command.index("--map-tokens") + 1] == "1024"
    assert {
        "--no-pretty",
        "--no-show-model-warnings",
        "--no-check-update",
        "--no-show-release-notes",
        "--no-gitignore",
        "--no-analytics",
        "--show-repo-map",
    } <= set(command)
    assert "--yes-always" not in command
    assert "--exit" not in command
    assert prepare["env"] == {"OPENAI_API_KEY": "${LITELLM_MASTER_KEY}"}


def test_aider_repomap_isolates_source_and_strips_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("def example():\n    return 1\n")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "example.py"], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fugue Test",
            "-c",
            "user.email=fugue@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source,
        check=True,
    )
    marker = "Here is the map.\n"
    raw = "startup warning\n" + marker + "example.py:\n  def example(): ..."
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            "Path('.aider.tags.cache.v4').mkdir(); "
            f"print({raw!r})"
        ),
    ]
    spec = ContextSystemSpec(
        id="aider-test",
        title="Aider test",
        provider="fugue.bench.context:AiderRepoMapContextProvider",
        version="test",
        capabilities=frozenset({"prepare"}),
        deliveries=frozenset({"portable"}),
        config={
            "prepare": {
                "command": command,
                "stdout_path": "REPO_MAP.md",
                "output_marker": marker,
            }
        },
    )
    output = tmp_path / "output"
    before = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    prepared = asyncio.run(
        AiderRepoMapContextProvider().prepare(
            spec,
            RepositorySnapshot("task", "fixture/repo", "HEAD", source),
            ContextRuntime(tmp_path, tmp_path / "cache", {}, output_dir=output),
        )
    )

    assert (prepared.path / "artifact" / "REPO_MAP.md").read_text() == (
        marker + "example.py:\n  def example(): ...\n"
    )
    after = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert after == before
    assert not (source / ".aider.tags.cache.v4").exists()


@pytest.mark.parametrize("raw", ["startup only\n", "marker\n"])
def test_aider_repomap_rejects_missing_or_empty_map(tmp_path: Path, raw: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("value = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "example.py"], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fugue Test",
            "-c",
            "user.email=fugue@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source,
        check=True,
    )
    spec = ContextSystemSpec(
        id="aider-test",
        title="Aider test",
        provider="fugue.bench.context:AiderRepoMapContextProvider",
        version="test",
        capabilities=frozenset({"prepare"}),
        deliveries=frozenset({"portable"}),
        config={
            "prepare": {
                "command": [sys.executable, "-c", f"print({raw!r})"],
                "stdout_path": "REPO_MAP.md",
                "output_marker": "marker\n",
            }
        },
    )

    with pytest.raises(RuntimeError, match="complete map"):
        asyncio.run(
            AiderRepoMapContextProvider().prepare(
                spec,
                RepositorySnapshot("task", "fixture/repo", "HEAD", source),
                ContextRuntime(
                    tmp_path,
                    tmp_path / "cache",
                    {},
                    output_dir=tmp_path / "output",
                ),
            )
        )


def test_isolated_command_does_not_mutate_the_source_checkout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("# Source\n")
    commit = _commit_repository(source)
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            "Path('openwiki').mkdir(); "
            "Path('openwiki/page.md').write_text('wiki'); "
            "Path('AGENTS.md').write_text('agents')"
        ),
    ]
    spec = ContextSystemSpec(
        id="isolated-command",
        title="Isolated command",
        provider="fugue.bench.context:IsolatedCommandContextProvider",
        version="test",
        capabilities=frozenset({"prepare"}),
        deliveries=frozenset({"portable"}),
        config={
            "prepare": {
                "command": command,
                "copy_paths": ["openwiki", "AGENTS.md"],
            }
        },
    )
    runtime = ContextRuntime(
        tmp_path,
        tmp_path / "cache",
        {},
        output_dir=tmp_path / "output",
    )

    provider = load_provider(spec)
    prepared = asyncio.run(
        provider.prepare(
            spec,
            RepositorySnapshot("task", "fixture/repo", commit, source),
            runtime,
        )
    )

    assert (prepared.path / "artifact" / "openwiki" / "page.md").read_text() == ("wiki")
    assert (prepared.path / "artifact" / "AGENTS.md").read_text() == "agents"
    assert not (source / "openwiki").exists()
    assert not (source / "AGENTS.md").exists()
    assert not subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_context_command_timeout_terminates_the_process_group(tmp_path: Path) -> None:
    pid_path = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os, time; "
            "from pathlib import Path; "
            f"Path({pid_path.as_posix()!r}).write_text(str(os.getpid())); "
            "time.sleep(30)"
        ),
    ]

    with pytest.raises(TimeoutError):
        asyncio.run(
            _run_context_command(
                command,
                cwd=tmp_path,
                env=dict(os.environ),
                capture_output=True,
                timeout_seconds=1.0,
            )
        )

    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_dense_artifact_is_shared_by_dense_and_hybrid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    versions = {
        "fastembed": "0.8.0",
        "onnxruntime": "1.23.2",
        "lancedb": "0.34.0",
    }
    monkeypatch.setattr(
        "fugue.bench.context.importlib.metadata.version", versions.__getitem__
    )
    chunks = [
        {
            "id": "src/app.py:1",
            "path": "src/app.py",
            "start_line": 1,
            "end_line": 1,
            "text": "value = 1",
        }
    ]
    builds = []

    def fake_build(artifact, values, model, dimensions):
        builds.append((values, model, dimensions))
        index = artifact / "lancedb"
        index.mkdir()
        (index / "data").write_text("stable-index")

    monkeypatch.setattr("fugue.bench.context._build_lance_index", fake_build)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    first_result = _materialize_dense_artifact(
        first,
        chunks,
        "BAAI/bge-small-en-v1.5",
        384,
        tmp_path / "cache",
    )
    second_result = _materialize_dense_artifact(
        second,
        chunks,
        "BAAI/bge-small-en-v1.5",
        384,
        tmp_path / "cache",
    )

    assert len(builds) == 1
    assert first_result["cache_hit"] is False
    assert second_result["cache_hit"] is True
    assert (first / "dense-artifact.json").read_bytes() == (
        second / "dense-artifact.json"
    ).read_bytes()
    assert (first / "lancedb" / "data").read_bytes() == (
        second / "lancedb" / "data"
    ).read_bytes()
    dense_key, _ = _dense_artifact_contract(chunks, "BAAI/bge-small-en-v1.5", 384)
    hybrid_key, _ = _dense_artifact_contract(chunks, "BAAI/bge-small-en-v1.5", 384)
    changed_key, _ = _dense_artifact_contract(chunks, "other/model", 384)
    assert dense_key == hybrid_key
    assert changed_key != dense_key


def test_embedding_model_override_is_used_by_the_provider_contract(
    tmp_path: Path,
) -> None:
    spec = get_context_system("rag-dense", tmp_path)
    runtime = ContextRuntime(
        tmp_path,
        tmp_path / "cache",
        {"FUGUE_EMBEDDING_MODEL": "local/override"},
    )

    assert _resolved_embedding_model(spec, runtime) == "local/override"


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


def test_rag_indexes_only_the_pinned_git_tree(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / ".fugue" / "cache" / "context" / "checkouts" / "repo"
    source = checkout / "src" / "app.py"
    source.parent.mkdir(parents=True)
    source.write_text("def answer():\n    return 42\n")
    commit = _commit_repository(checkout)
    source.write_text("def answer():\n    return 'dirty'\n")
    (checkout / ".fugue-snapshot.json").write_text('{"task": "leak"}\n')
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n")
    (checkout / "escape.py").symlink_to(outside)
    snapshot = RepositorySnapshot(
        task_id="fixture__task",
        repo="fixture/repo",
        commit=commit,
        checkout=checkout,
    )
    runtime = ContextRuntime(tmp_path, tmp_path / "context-cache", {})

    prepared = asyncio.run(
        prepare_context(get_context_system("rag-bm25", tmp_path), snapshot, runtime)
    )

    chunks = (prepared.path / "artifact" / "chunks.jsonl").read_text().splitlines()
    assert len(chunks) == 1
    [chunk] = [json.loads(line) for line in chunks]
    assert chunk["path"] == "src/app.py"
    assert "return 42" in chunk["text"]
    assert "dirty" not in chunk["text"]
    assert prepared.metrics["files"] == 1

    other_task = RepositorySnapshot(
        task_id="another-task",
        repo=snapshot.repo,
        commit=commit,
        checkout=checkout,
        dataset_id="another-dataset",
    )
    assert _repository_chunks(
        snapshot, lines_per_chunk=80, overlap=20
    ) == _repository_chunks(other_task, lines_per_chunk=80, overlap=20)


def test_rag_rejects_empty_repository_instead_of_publishing_empty_index(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "empty"
    checkout.mkdir()
    commit = _commit_repository(checkout)
    snapshot = RepositorySnapshot(
        task_id="fixture__task",
        repo="fixture/repo",
        commit=commit,
        checkout=checkout,
    )
    spec = get_context_system("rag-bm25", tmp_path)
    runtime = ContextRuntime(tmp_path, tmp_path / "context-cache", {})

    with pytest.raises(ValueError, match="no indexable repository text"):
        asyncio.run(prepare_context(spec, snapshot, runtime))

    assert not (runtime.cache_root / context_cache_key(spec, snapshot)).exists()


def test_managed_repository_copy_rejects_escaping_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    (source / "escape.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="absolute repository symlink"):
        _copy_repository_snapshot(source, tmp_path / "target", ignored=())


def test_latmd_preparation_uses_the_pinned_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "README.md").write_text("fixture\n")
    commit = _commit_repository(checkout)
    output = tmp_path / "output"
    spec = get_context_system("latmd", Path(__file__).resolve().parents[1])
    runtime = ContextRuntime(
        tmp_path,
        tmp_path / "cache",
        {"LAT_LLM_KEY": "private", "FUGUE_ENABLE_EXPERIMENTAL_LATMD": "true"},
        output_dir=output,
    )
    calls: list[str] = []

    def fake_prepare(system_id, *, repo_root, artifact, env):
        calls.append(system_id)
        (artifact / "repository" / "lat.md").mkdir()

    monkeypatch.setattr(
        "fugue.bench.context.prepare_runtime_repository", fake_prepare
    )
    prepared = asyncio.run(
        load_provider(spec).prepare(
            spec,
            RepositorySnapshot("task", "repo", commit, checkout, "dataset"),
            runtime,
        )
    )

    assert calls == ["latmd"]
    assert (prepared.path / "artifact/repository/lat.md/episodes.md").is_file()
    assert (prepared.path / "artifact/FUGUE_NATIVE_MCP.md").is_file()


def test_gitnexus_artifact_contains_container_valid_repository_and_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "src").mkdir(parents=True)
    (checkout / "src" / "app.py").write_text("print('ok')\n")
    commit = _commit_repository(checkout)
    output = tmp_path / "output"
    spec = get_context_system("gitnexus", tmp_path)
    runtime = ContextRuntime(
        tmp_path,
        tmp_path / "cache",
        {"FUGUE_LICENSE_APPROVED_GITNEXUS": "true"},
        output_dir=output,
    )

    def fake_prepare_runtime(system_id, *, repo_root, artifact, env):
        assert system_id == "gitnexus"
        repository = artifact / "repository"
        (repository / ".gitnexus").mkdir()
        (repository / ".gitnexus" / "graph.json").write_text("{}")
        registry = artifact / "home" / ".gitnexus" / "registry.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({"/workspace/repository": {"ready": True}}))

    monkeypatch.setattr(
        "fugue.bench.context.prepare_runtime_repository",
        fake_prepare_runtime,
    )
    provider = load_provider(spec)
    prepared = asyncio.run(
        provider.prepare(
            spec,
            RepositorySnapshot("task", "repo", commit, checkout, "dataset"),
            runtime,
        )
    )

    repository = prepared.path / "artifact" / "repository"
    assert (repository / "src" / "app.py").is_file()
    assert (repository / ".gitnexus" / "graph.json").is_file()
    registry = prepared.path / "artifact" / "home" / ".gitnexus" / "registry.json"
    assert "/workspace/repository" in registry.read_text()
    assert checkout.as_posix() not in registry.read_text()
    instruction = prepared.path / "artifact" / "FUGUE_NATIVE_MCP.md"
    assert "/workspace/repository" in instruction.read_text()
    binding = asyncio.run(
        provider.bind(
            spec,
            prepared,
            TrialContext("experiment", "workload", "task", "harness"),
            runtime,
            "native_mcp",
        )
    )
    assert binding.extra_instruction_paths == (instruction,)
    assert not (checkout / ".gitnexus").exists()


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


def test_direct_workload_rows_append_safely_across_workers(tmp_path: Path) -> None:
    batches = [
        [{"worker": worker, "row": row} for row in range(100)] for worker in range(8)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda rows: _write_rows(tmp_path, "run", rows), batches))

    assert len(set(paths)) == 1
    values = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert len(values) == 800
    assert {(value["worker"], value["row"]) for value in values} == {
        (worker, row) for worker in range(8) for row in range(100)
    }


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


def test_portable_context_client_probes_and_records_bounded_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def request(url: str, body: dict[str, object] | None = None) -> dict[str, object]:
        if body is None:
            return {"ok": True, "context_system_id": "rag-bm25"}
        return {
            "ok": True,
            "context_system_id": "rag-bm25",
            "hits": [
                {
                    "path": "src/app.py",
                    "score": 1.0,
                    "text": body["query"],
                }
            ],
            "metrics": {"result_count": 1, "query_latency_ms": 2.0},
        }

    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("FUGUE_CONTEXT_EVENTS_PATH", events.as_posix())
    monkeypatch.setattr(context_client, "_request", request)

    assert context_client.main(["probe"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert (
        context_client.main(["query", "--text", "find the parser", "--top-k", "3"]) == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["hits"][0]["path"] == "src/app.py"
    [event] = [json.loads(line) for line in events.read_text().splitlines()]
    assert event["layer"] == "portable_client"
    assert event["metrics"]["result_count"] == 1


@pytest.mark.skipif(
    os.environ.get("FUGUE_RUN_PORTABLE_CONTEXT_INTEGRATION") != "1",
    reason="requires Docker plus configured W&B Inference and Weave credentials",
)
def test_portable_bm25_registers_across_all_four_harnesses(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    jobs_dir = tmp_path / "portable-bm25-all-harnesses"
    command = [
        sys.executable,
        "-m",
        "fugue.bench.cli",
        "run",
        "repo-memory-impact",
        "--preset",
        "smoke",
        "--workloads",
        "coding",
        "--systems",
        "none,rag-bm25",
        "--harnesses",
        "hermes,openclaw,claude-code,codex",
        "--model",
        "wandb/zai-org/GLM-5.2",
        "--jobs-dir",
        jobs_dir.as_posix(),
        "--run-name",
        "portable-context-integration",
        "-k",
        "1",
        "-n",
        "2",
        "-l",
        "1",
        "--trace-content",
        "full",
        "--json",
        "--repo-root",
        repo_root.as_posix(),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=7_200,
        check=False,
    )
    assert completed.returncode in {0, 1}, completed.stderr[-4_000:]
    summary = json.loads(completed.stdout)
    assert summary["status"] in {"passed", "failed"}
    assert summary["passed"] + summary["failed"] == 8

    rows = [
        row
        for row in export_rows([jobs_dir], env=os.environ)
        if row.get("record_type") == "trial"
    ]
    assert len(rows) == 8
    assert {row["harness"] for row in rows} == {
        "hermes",
        "openclaw",
        "claude-code",
        "codex",
    }
    rag_rows = [row for row in rows if row["context_system_id"] == "rag-bm25"]
    assert len(rag_rows) == 4
    assert all(row["context_delivery"] == "portable" for row in rag_rows)
    assert all(row["context_registered"] is True for row in rag_rows)
    assert all(row["runtime_equivalent"] is True for row in rows)
