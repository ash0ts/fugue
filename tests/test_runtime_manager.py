from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from fugue.bench import runtime_manager

VECTOR_CONFIG = {
    "retrieval_mode": "hybrid_vector",
    "embedding_model": "Snowflake/snowflake-arctic-embed-xs",
    "embedding_revision": "d8c86521100d3556476a063fc2342036d45c106f",
    "embedding_dimensions": 384,
    "vector_required": True,
}


def test_managed_runtime_catalog_is_pinned_and_install_free_at_trial_time() -> None:
    assert set(runtime_manager.RUNTIMES) == {
        "gitnexus",
        "codegraph",
        "semble",
        "project-rag",
        "latmd",
    }
    assert runtime_manager.RUNTIMES["codegraph"].version.endswith("@0.9.0")
    semble = runtime_manager.RUNTIMES["semble"]
    assert semble.upstream_command[:2] == ("/opt/gateway/bin/python", "-c")
    assert "semble.mcp import serve" in semble.upstream_command[2]
    assert dict(semble.asset_integrities) == {
        "minishlab/potion-code-16M-v2": (
            "git:e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b"
        ),
        "tree-sitter-language-pack": (
            "version:1.6.2;languages:bash,c,cpp,csharp,css,dockerfile,go,html,"
            "java,javascript,json,markdown,php,pkl,python,ruby,rust,toml,tsx,"
            "typescript,yaml"
        ),
    }
    assert dict(semble.runtime_env) == {
        "SEMBLE_MODEL_NAME": "/opt/semble-model",
        "SEMBLE_TREE_SITTER_CACHE": "/opt/tree-sitter-languages",
        "HF_HUB_OFFLINE": "1",
    }
    assert runtime_manager.RUNTIMES["project-rag"].version.endswith(
        "d5abf98a48b60d35b73745e47e1aacca3963a6f0"
    )
    assert runtime_manager.RUNTIMES["latmd"].prepare_command == ("lat", "init")
    assert runtime_manager.RUNTIMES["gitnexus"].repository_state_paths == (".gitnexus",)
    assert runtime_manager.RUNTIMES["codegraph"].repository_state_paths == (
        ".codegraph",
    )
    assert runtime_manager.RUNTIMES["latmd"].repository_state_paths == ("lat.md",)
    project_rag = runtime_manager.RUNTIMES["project-rag"]
    assert dict(project_rag.asset_integrities) == {
        "Qdrant/all-MiniLM-L6-v2-onnx": ("git:5f1b8cd78bc4fb444dd171e59b18f3a3af89a079")
    }
    assert dict(project_rag.runtime_env)["PROJECT_RAG_LANCEDB_PATH"] == (
        "/workspace/state/lancedb"
    )
    assert dict(project_rag.runtime_env)["RUST_LOG"] == "off"
    for spec in runtime_manager.RUNTIMES.values():
        assert len(spec.recipe_sha256) == 64
        assert "latest" not in spec.dockerfile
        assert spec.upstream_command[0] not in {"npx", "uvx", "cargo"}
        assert spec.entrypoint == ("/opt/fugue/start-gateway",)
        assert spec.health_check
        assert spec.network_policy == "share_cell_network"
        assert spec.repository_mount == "/workspace/repository"
        assert spec.state_mount == "/workspace/state"
    assert dict(runtime_manager.RUNTIMES["gitnexus"].runtime_env) == {
        "GITNEXUS_HOME": "/workspace/state/home/.gitnexus",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "FUGUE_GITNEXUS_MODEL_DIGEST": (
            "cf2698d30ff05da02c70a088313bad56e5c2f401d734cb24a8390d446111936c"
        ),
    }
    gitnexus = runtime_manager.RUNTIMES["gitnexus"]
    assert gitnexus.version == "gitnexus@1.6.3+fugue-vector.6"
    assert "ONNXRUNTIME_NODE_INSTALL=skip" in gitnexus.dockerfile
    assert "npm ci --ignore-scripts" in gitnexus.dockerfile
    assert "allowRemoteModels" not in gitnexus.dockerfile
    assert dict(gitnexus.asset_integrities) == {
        "Snowflake/snowflake-arctic-embed-xs": (
            "git:d8c86521100d3556476a063fc2342036d45c106f"
        ),
        "embedding_dimensions": "384",
        "onnxruntime-node": "version:1.24.3;cpu-only",
        "lexical_index": "flat-bm25-v1",
        "vector_index": "flat-cosine-v1;max-nodes:50000",
    }


def test_prepare_runtime_writes_image_identity_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
                            "RepoDigests": ["example@sha256:" + "b" * 64],
                            "Architecture": "arm64",
                            "Os": "linux",
                        }
                    ]
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime_manager.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(runtime_manager.subprocess, "run", run)
    monkeypatch.setattr(
        runtime_manager,
        "docker_build_command",
        lambda *args: ["docker", "build", "--provenance=false", *args],
    )

    lock = runtime_manager.prepare_runtime("gitnexus", repo_root=tmp_path)

    assert lock["image_id"] == "sha256:" + "a" * 64
    assert lock["architecture"] == "arm64"
    assert lock["network_policy"] == "share_cell_network"
    assert commands[0][:4] == ["docker", "build", "--provenance=false", "--pull"]
    stored = runtime_manager.read_runtime_lock("gitnexus", tmp_path)
    assert stored == lock
    build = tmp_path / runtime_manager.RUNTIME_ROOT / "gitnexus/build"
    assert "npm ci --ignore-scripts" in (build / "Dockerfile").read_text()
    assert (
        json.loads((build / "package.json").read_text())["dependencies"]["gitnexus"]
        == "1.6.3"
    )


def test_runtime_compose_uses_isolated_sidecar_and_read_only_repository(
    tmp_path: Path,
) -> None:
    spec = runtime_manager.RUNTIMES["semble"]
    root = tmp_path / runtime_manager.RUNTIME_ROOT / "semble"
    root.mkdir(parents=True)
    (root / "runtime-lock.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "system_id": "semble",
                "version": spec.version,
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image,
                "image_id": "sha256:" + "a" * 64,
                "upstream_command": list(spec.upstream_command),
            }
        )
    )
    repository = tmp_path / "repository"
    repository.mkdir()

    path, server, descriptor = runtime_manager.render_runtime_compose(
        "semble",
        repo_root=tmp_path,
        artifact=repository,
        runtime_root=tmp_path / ".fugue/runtime/run-a",
        job_name="job-a",
        env_names=(),
        write=True,
    )

    compose = runtime_manager.yaml.safe_load(path.read_text())
    service = compose["services"]["fugue-semble"]
    assert service["network_mode"] == "service:main"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["volumes"][:2] == [
        f"{repository.resolve().as_posix()}:/fugue-context:ro",
        (
            f"{(repository / 'repository').resolve().as_posix()}:"
            "/workspace/repository:ro"
        ),
    ]
    evidence_mount = service["volumes"][2]
    assert evidence_mount["target"] == "/fugue-evidence"
    assert evidence_mount["read_only"] is False
    assert evidence_mount["bind"]["create_host_path"] is False
    assert Path(evidence_mount["source"]).is_dir()
    assert service["environment"]["FUGUE_GATEWAY_EVENT_LOG"] == (
        "/fugue-evidence/context-gateway.jsonl"
    )
    assert service["tmpfs"] == [
        "/tmp:rw,noexec,nosuid,size=64m",
        "/workspace/state:rw,noexec,nosuid,size=2g",
    ]
    assert service["environment"]["FUGUE_REPOSITORY_STATE_PATHS"] == ""
    assert server == {
        "name": "semble",
        "transport": "streamable-http",
        "url": "http://127.0.0.1:8765/mcp",
    }
    assert descriptor["image_id"] == "sha256:" + "a" * 64


def test_repository_preparation_uses_the_active_runtime_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = runtime_manager.RUNTIMES["gitnexus"]
    root = tmp_path / runtime_manager.RUNTIME_ROOT / "gitnexus"
    root.mkdir(parents=True)
    (root / "runtime-lock.json").write_text(
        json.dumps(
            {
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image,
                "image_id": "sha256:" + "a" * 64,
            }
        )
    )
    artifact = tmp_path / "artifact"
    (artifact / "repository").mkdir(parents=True)
    calls: list[tuple[list[str], dict]] = []

    def run(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "index output", "diagnostic")

    monkeypatch.setattr(runtime_manager.subprocess, "run", run)

    runtime_manager.prepare_runtime_repository(
        "gitnexus",
        repo_root=tmp_path,
        artifact=artifact,
        env={},
        config={"retrieval_mode": "bm25"},
    )

    command, kwargs = calls[0]
    assert any(value.endswith(",dst=/workspace/repository") for value in command)
    assert any(value.endswith(",dst=/workspace/state/home") for value in command)
    assert command[-1] == "/workspace/repository"
    assert "GITNEXUS_HOME=/workspace/state/home/.gitnexus" in command
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_repository_preparation_reports_captured_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = runtime_manager.RUNTIMES["gitnexus"]
    root = tmp_path / runtime_manager.RUNTIME_ROOT / "gitnexus"
    root.mkdir(parents=True)
    (root / "runtime-lock.json").write_text(
        json.dumps(
            {
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image,
                "image_id": "sha256:" + "a" * 64,
            }
        )
    )
    artifact = tmp_path / "artifact"
    (artifact / "repository").mkdir(parents=True)

    def fail(command: list[str], **kwargs):
        raise subprocess.CalledProcessError(
            17,
            command,
            output="index output",
            stderr="index diagnostic",
        )

    monkeypatch.setattr(runtime_manager.subprocess, "run", fail)

    with pytest.raises(
        RuntimeError,
        match=(
            "(?s)gitnexus repository preparation failed: "
            "index output.*index diagnostic"
        ),
    ):
        runtime_manager.prepare_runtime_repository(
            "gitnexus",
            repo_root=tmp_path,
            artifact=artifact,
            env={},
            config={"retrieval_mode": "bm25"},
        )


def test_gitnexus_hybrid_preparation_is_offline_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = runtime_manager.RUNTIMES["gitnexus"]
    root = tmp_path / runtime_manager.RUNTIME_ROOT / "gitnexus"
    root.mkdir(parents=True)
    (root / "runtime-lock.json").write_text(
        json.dumps(
            {
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image,
                "image_id": "sha256:" + "a" * 64,
            }
        )
    )
    artifact = tmp_path / "artifact"
    (artifact / "repository/.gitnexus").mkdir(parents=True)
    (artifact / "home").mkdir()
    (artifact / "repository/.gitnexus/meta.json").write_text(
        json.dumps({"stats": {"nodes": 10, "embeddings": 7}})
    )
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        commands.append(command)
        if "query" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                '{"filePath":"src/relay/amber_lantern.py"}',
                'FUGUE_GITNEXUS_VECTOR {"vector_search_succeeded":true}',
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime_manager.subprocess, "run", run)
    runtime_manager.prepare_runtime_repository(
        "gitnexus",
        repo_root=tmp_path,
        artifact=artifact,
        env={},
        config=VECTOR_CONFIG,
        semantic_probe={
            "query": "hierarchical cleanup upstream grant invalid",
            "expected_path": "src/relay/amber_lantern.py",
        },
    )
    graph_analyze, vector_analyze, probe = commands
    for command in commands:
        assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "--network" in graph_analyze and "none" in graph_analyze
    assert "--embeddings" not in graph_analyze
    assert "--embeddings" in vector_analyze
    assert "FUGUE_GITNEXUS_VECTOR_REQUIRED=1" in vector_analyze
    assert "query" in probe
    assert "hierarchical cleanup upstream grant invalid" in probe
    assert "--network" in probe and "none" in probe


def test_gitnexus_hybrid_rejects_large_graph_before_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = runtime_manager.RUNTIMES["gitnexus"]
    root = tmp_path / runtime_manager.RUNTIME_ROOT / "gitnexus"
    root.mkdir(parents=True)
    (root / "runtime-lock.json").write_text(
        json.dumps(
            {
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image,
                "image_id": "sha256:" + "a" * 64,
            }
        )
    )
    artifact = tmp_path / "artifact"
    (artifact / "repository/.gitnexus").mkdir(parents=True)
    (artifact / "repository/.gitnexus/meta.json").write_text(
        json.dumps({"stats": {"nodes": 50_001, "embeddings": 0}})
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        runtime_manager.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )

    with pytest.raises(RuntimeError, match="safety limit is 50000"):
        runtime_manager.prepare_runtime_repository(
            "gitnexus",
            repo_root=tmp_path,
            artifact=artifact,
            env={},
            config=VECTOR_CONFIG,
        )

    assert len(commands) == 1
    assert "--embeddings" not in commands[0]


def test_gitnexus_direct_query_preserves_variant_and_vector_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = runtime_manager.RUNTIMES["gitnexus"]
    root = tmp_path / runtime_manager.RUNTIME_ROOT / "gitnexus"
    root.mkdir(parents=True)
    (root / "runtime-lock.json").write_text(
        json.dumps(
            {
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image,
                "image_id": "sha256:" + "a" * 64,
            }
        )
    )
    artifact = tmp_path / "artifact"
    (artifact / "repository").mkdir(parents=True)
    (artifact / "home").mkdir()
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "definitions": [
                        {"filePath": "src/relay/amber_lantern.py", "name": "withdraw"}
                    ]
                }
            ),
            "\n".join(
                (
                    'FUGUE_GITNEXUS_VECTOR {"vector_search_attempted":true}',
                    'FUGUE_GITNEXUS_VECTOR {"vector_search_succeeded":true,'
                    '"semantic_result_count":4,"bm25_result_count":0,'
                    '"model_digest":"model-sha","query_latency_ms":12.5}',
                )
            ),
        )

    monkeypatch.setattr(runtime_manager.subprocess, "run", run)
    payload, telemetry = runtime_manager.query_gitnexus(
        repo_root=tmp_path,
        artifact=artifact,
        config=VECTOR_CONFIG,
        query="hierarchical cleanup upstream grant invalid",
        top_k=3,
    )

    assert payload["definitions"][0]["filePath"].endswith("amber_lantern.py")
    assert telemetry == {
        "retrieval_mode": "hybrid_vector",
        "vector_search_attempted": True,
        "vector_search_succeeded": True,
        "semantic_result_count": 4,
        "bm25_result_count": 0,
        "vector_model_digest": "model-sha",
        "vector_query_latency_ms": 12.5,
    }
    assert commands[0][commands[0].index("--network") + 1] == "none"
    assert "FUGUE_GITNEXUS_VECTOR_REQUIRED=1" in commands[0]


def test_gitnexus_direct_hybrid_query_rejects_silent_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = runtime_manager.RUNTIMES["gitnexus"]
    root = tmp_path / runtime_manager.RUNTIME_ROOT / "gitnexus"
    root.mkdir(parents=True)
    (root / "runtime-lock.json").write_text(
        json.dumps(
            {
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image,
                "image_id": "sha256:" + "a" * 64,
            }
        )
    )
    artifact = tmp_path / "artifact"
    (artifact / "repository").mkdir(parents=True)
    (artifact / "home").mkdir()
    monkeypatch.setattr(
        runtime_manager.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, '{"definitions":[]}', ""
        ),
    )

    with pytest.raises(RuntimeError, match="did not execute vector retrieval"):
        runtime_manager.query_gitnexus(
            repo_root=tmp_path,
            artifact=artifact,
            config=VECTOR_CONFIG,
            query="semantic only",
            top_k=3,
        )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"retrieval_mode": "vector"},
        {"retrieval_mode": "hybrid_vector", "vector_required": False},
    ],
)
def test_gitnexus_retrieval_contract_rejects_ambiguous_modes(config) -> None:
    with pytest.raises(ValueError, match="GitNexus"):
        runtime_manager.gitnexus_retrieval_mode(config)


def test_mutable_adapter_index_is_isolated_from_the_read_only_repository(
    tmp_path: Path,
) -> None:
    spec = runtime_manager.RUNTIMES["gitnexus"]
    root = tmp_path / runtime_manager.RUNTIME_ROOT / "gitnexus"
    root.mkdir(parents=True)
    (root / "runtime-lock.json").write_text(
        json.dumps(
            {
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image,
                "image_id": "sha256:" + "a" * 64,
            }
        )
    )
    artifact = tmp_path / "artifact"
    (artifact / "repository/.gitnexus").mkdir(parents=True)

    _, _, _ = runtime_manager.render_runtime_compose(
        "gitnexus",
        repo_root=tmp_path,
        artifact=artifact,
        runtime_root=tmp_path / "runtime",
        job_name="job",
        env_names=(),
        write=True,
        context_config={"retrieval_mode": "bm25"},
    )
    compose = runtime_manager.yaml.safe_load(
        (tmp_path / "runtime/context-runtimes/job.yaml").read_text()
    )
    service = compose["services"]["fugue-gitnexus"]
    assert service["volumes"][1].endswith(":/workspace/repository:ro")
    assert service["tmpfs"][-1] == (
        "/workspace/repository/.gitnexus:rw,noexec,nosuid,size=2g"
    )
    assert service["environment"]["FUGUE_REPOSITORY_STATE_PATHS"] == ".gitnexus"


def test_managed_runtime_command_passes_secret_by_environment_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = runtime_manager.RUNTIMES["latmd"]
    root = tmp_path / runtime_manager.RUNTIME_ROOT / "latmd"
    root.mkdir(parents=True)
    (root / "runtime-lock.json").write_text(
        json.dumps(
            {
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image,
                "image_id": "sha256:" + "a" * 64,
            }
        )
    )
    repository = tmp_path / "repository"
    repository.mkdir()
    calls: list[tuple[list[str], dict]] = []

    def run(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "result", "")

    monkeypatch.setattr(runtime_manager.subprocess, "run", run)
    result = runtime_manager.run_runtime_command(
        "latmd",
        repo_root=tmp_path,
        repository=repository,
        env={"LAT_LLM_KEY": "private-value"},
        command=("lat", "search", "query"),
    )

    command, kwargs = calls[0]
    assert result.stdout == "result"
    assert "private-value" not in command
    assert command[command.index("--env") + 1] == "HOME=/workspace/state/home"
    assert command[command.index("--env", command.index("--env") + 1) + 1] == (
        "LAT_LLM_KEY"
    )
    assert kwargs["env"]["LAT_LLM_KEY"] == "private-value"
    assert command[-3:] == ["sha256:" + "a" * 64, "search", "query"]
