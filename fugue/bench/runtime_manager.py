from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from fugue.bench.files import atomic_write_json
from fugue.bench.files import inspect_docker_image as _inspect_image

RUNTIME_ROOT = Path(".fugue/runtime/context-runtimes")
GATEWAY_PORT = 8765
GITNEXUS_VECTOR_MODE = "hybrid_vector"
GITNEXUS_VECTOR_DIMENSIONS = 384


@dataclass(frozen=True)
class ManagedMCPRuntimeSpec:
    system_id: str
    version: str
    dockerfile: str
    upstream_command: tuple[str, ...]
    package_integrity: str | None = None
    platform_integrities: tuple[tuple[str, str], ...] = ()
    asset_integrities: tuple[tuple[str, str], ...] = ()
    prepare_command: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ("amd64", "arm64")
    entrypoint: tuple[str, ...] = ("/opt/fugue/start-gateway",)
    health_check: tuple[str, ...] = (
        "/opt/gateway/bin/python",
        "-c",
        "import socket;s=socket.create_connection(('127.0.0.1',8765),2);s.close()",
    )
    network_policy: str = "share_cell_network"
    repository_mount: str = "/workspace/repository"
    state_mount: str = "/workspace/state"
    repository_state_paths: tuple[str, ...] = ()
    runtime_env: tuple[tuple[str, str], ...] = ()
    probe_command: tuple[str, ...] = ()

    @property
    def recipe_sha256(self) -> str:
        gateway = Path(__file__).resolve().parents[1] / "mcp_gateway.py"
        build_assets = _runtime_build_assets(self.system_id)
        return _digest(
            {
                "spec": asdict(self),
                "gateway_sha256": hashlib.sha256(gateway.read_bytes()).hexdigest(),
                "entrypoint_sha256": hashlib.sha256(
                    _gateway_entrypoint().encode()
                ).hexdigest(),
                "build_assets": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in build_assets
                },
            }
        )

    @property
    def image(self) -> str:
        return f"fugue-context-{self.system_id}:{self.recipe_sha256[:12]}"

    @property
    def install_probe_command(self) -> tuple[str, ...]:
        return self.probe_command or (*self.upstream_command, "--help")


_GATEWAY_INSTALL = (
    "python3 -m venv /opt/gateway && "
    "/opt/gateway/bin/pip install --no-cache-dir "
    "mcp==1.28.1 uvicorn==0.41.0 starlette==0.52.1"
)

_NODE_IMAGE = (
    "node:22.17.1-bookworm-slim@"
    "sha256:2fa754a9ba4d7adbd2a51d182eaabbe355c82b673624035a38c0d42b08724854"
)
_PYTHON_IMAGE = (
    "python:3.12.11-slim-bookworm@"
    "sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
)
_PROJECT_RAG_RUST_IMAGE = (
    "rust:1.88-trixie@"
    "sha256:f2a17efbe58b00470be6e73dbea79705a789ee20ee8de2dcdaf73c0c6091f1db"
)
_PROJECT_RAG_PYTHON_IMAGE = (
    "python:3.12.11-slim-trixie@"
    "sha256:47ae396f09c1303b8653019811a8498470603d7ffefc29cb07c88f1f8cb3d19f"
)
_SEMBLE_LANGUAGES = (
    "bash,c,cpp,csharp,css,dockerfile,go,html,java,javascript,json,markdown,php,"
    "pkl,python,ruby,rust,toml,tsx,typescript,yaml"
)
_SEMBLE_COMMAND = (
    "/opt/gateway/bin/python",
    "-c",
    "import asyncio, os; import tree_sitter_language_pack as pack; "
    "pack.configure(cache_dir=os.environ['SEMBLE_TREE_SITTER_CACHE']); "
    "from semble.mcp import serve; asyncio.run(serve())",
)


def _runtime_build_assets(system_id: str) -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2] / "configs/fugue/runtime" / system_id
    if not root.is_dir():
        return ()
    return tuple(path for path in sorted(root.iterdir()) if path.is_file())


def _node_runtime(
    package: str,
    command: tuple[str, ...],
    *,
    ignore_scripts: bool = True,
) -> str:
    install_flags = "--ignore-scripts " if ignore_scripts else ""
    build_packages = "" if ignore_scripts else " build-essential git"
    return f"""FROM {_NODE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates{build_packages} && rm -rf /var/lib/apt/lists/*
RUN {_GATEWAY_INSTALL}
RUN npm install --global {install_flags}{package}
COPY mcp_gateway.py /opt/fugue/mcp_gateway.py
COPY start-gateway /opt/fugue/start-gateway
RUN chmod 0555 /opt/fugue/start-gateway
ENTRYPOINT [\"/opt/fugue/start-gateway\"]
CMD {json.dumps(list(command))}
"""


def _gitnexus_runtime() -> str:
    revision = "d8c86521100d3556476a063fc2342036d45c106f"
    model = "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve"
    return f"""FROM {_NODE_IMAGE}
ARG TARGETARCH
ENV ONNXRUNTIME_NODE_INSTALL=skip
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl build-essential git && rm -rf /var/lib/apt/lists/*
RUN {_GATEWAY_INSTALL}
WORKDIR /opt/gitnexus-runtime
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
RUN npm_config_nodedir=/usr/local npm rebuild --foreground-scripts tree-sitter tree-sitter-c tree-sitter-c-sharp tree-sitter-cpp tree-sitter-go tree-sitter-java tree-sitter-javascript tree-sitter-php tree-sitter-python tree-sitter-ruby tree-sitter-rust tree-sitter-typescript && node -e "for (const p of ['tree-sitter','tree-sitter-c','tree-sitter-c-sharp','tree-sitter-cpp','tree-sitter-go','tree-sitter-java','tree-sitter-javascript','tree-sitter-php','tree-sitter-python','tree-sitter-ruby','tree-sitter-rust','tree-sitter-typescript']) require(p)"
RUN set -eu; case "$TARGETARCH" in \
      amd64) ladybug=4d2d045404f8e31cd6d5bd8aec0adac64ce5bef8c904ab79408a5a3dbc8dce56; ortlib=92ba8e44d31b80a34f5e805076a044122c3be1fffe0d88ed790de7267b0b2682; ortbinding=a7191fbfe8045d4d5d312e5ecd8f5dd6970e804dd340c98be8a0af8daf7908cc; platform=x64 ;; \
      arm64) ladybug=33cf6cb1f188cac902412fc60f7fb1e79c7e1a92c34b373e6679ef8c02ff620b; ortlib=33032cb5b91c12edbf028d7c69fa40a31848d66c0095448758af237d5e0b07db; ortbinding=39c8f903505611115a62d8527b2c2df4bb5e27a35cfd81d1f2082eeb59518a4e; platform=arm64 ;; \
      *) echo "unsupported GitNexus architecture: $TARGETARCH" >&2; exit 2 ;; \
    esac; \
    source="node_modules/@ladybugdb/core-linux-$platform/lbugjs.node"; \
    echo "$ladybug  $source" | sha256sum -c -; \
    cp "$source" node_modules/@ladybugdb/core/lbugjs.node; \
    ort=node_modules/onnxruntime-node/bin/napi-v6/linux/$platform; \
    echo "$ortlib  $ort/libonnxruntime.so.1" | sha256sum -c -; \
    echo "$ortbinding  $ort/onnxruntime_binding.node" | sha256sum -c -
COPY patch-runtime.mjs ./
RUN node patch-runtime.mjs /opt/gitnexus-runtime/node_modules/gitnexus
RUN ln -s /opt/gitnexus-runtime/node_modules/gitnexus/dist/cli/index.js /usr/local/bin/gitnexus
RUN set -eu; root=/opt/gitnexus-models/Snowflake/snowflake-arctic-embed-xs; mkdir -p "$root/onnx"; \
    curl -fsSL "{model}/{revision}/config.json" -o "$root/config.json"; \
    curl -fsSL "{model}/{revision}/onnx/model.onnx" -o "$root/onnx/model.onnx"; \
    curl -fsSL "{model}/{revision}/tokenizer.json" -o "$root/tokenizer.json"; \
    curl -fsSL "{model}/{revision}/tokenizer_config.json" -o "$root/tokenizer_config.json"; \
    echo "d7d071046ab952af96b7abad788db7ab3fc997b465e1b9914ff39707092254ec  $root/config.json" | sha256sum -c -; \
    echo "cf2698d30ff05da02c70a088313bad56e5c2f401d734cb24a8390d446111936c  $root/onnx/model.onnx" | sha256sum -c -; \
    echo "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854  $root/tokenizer.json" | sha256sum -c -; \
    echo "9ca59277519f6e3692c8685e26b94d4afca2d5438deff66483db495e48735810  $root/tokenizer_config.json" | sha256sum -c -
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ORT_LOG_LEVEL=3
COPY mcp_gateway.py /opt/fugue/mcp_gateway.py
COPY start-gateway /opt/fugue/start-gateway
RUN chmod 0555 /opt/fugue/start-gateway /usr/local/bin/gitnexus
ENTRYPOINT [\"/opt/fugue/start-gateway\"]
CMD [\"gitnexus\", \"mcp\"]
"""


def _python_runtime(
    package: str,
    command: tuple[str, ...],
    *,
    post_install: str = "",
) -> str:
    post_install_step = f"RUN {post_install}\n" if post_install else ""
    return f"""FROM {_PYTHON_IMAGE}
RUN python -m venv /opt/gateway && /opt/gateway/bin/pip install --no-cache-dir mcp==1.28.1 uvicorn==0.41.0 starlette==0.52.1
RUN /opt/gateway/bin/pip install --no-cache-dir {package}
{post_install_step}COPY mcp_gateway.py /opt/fugue/mcp_gateway.py
COPY start-gateway /opt/fugue/start-gateway
RUN chmod 0555 /opt/fugue/start-gateway
ENTRYPOINT [\"/opt/fugue/start-gateway\"]
CMD {json.dumps(list(command))}
"""


def _codegraph_runtime() -> str:
    return f"""FROM {_PYTHON_IMAGE}
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/gateway && /opt/gateway/bin/pip install --no-cache-dir mcp==1.28.1 uvicorn==0.41.0 starlette==0.52.1
RUN set -eu; case "$TARGETARCH" in \
      amd64) archive=codegraph-linux-x64.tar.gz; digest=5f777f6005c6b6c75137cecd112911278e31f86c63f0cafaef367e74294eb629 ;; \
      arm64) archive=codegraph-linux-arm64.tar.gz; digest=89e9eff1165d8251157640658d260aa0a95a8a6ee1da9c64f4cd2fa4136ba649 ;; \
      *) echo "unsupported CodeGraph architecture: $TARGETARCH" >&2; exit 2 ;; \
    esac; \
    curl -fsSLo /tmp/codegraph.tar.gz "https://github.com/colbymchenry/codegraph/releases/download/v0.9.0/$archive"; \
    echo "$digest  /tmp/codegraph.tar.gz" | sha256sum -c -; \
    mkdir /opt/codegraph; tar -xzf /tmp/codegraph.tar.gz -C /opt/codegraph --strip-components=1; \
    rm /tmp/codegraph.tar.gz
COPY mcp_gateway.py /opt/fugue/mcp_gateway.py
COPY start-gateway /opt/fugue/start-gateway
RUN chmod 0555 /opt/fugue/start-gateway /opt/codegraph/bin/codegraph
ENTRYPOINT [\"/opt/fugue/start-gateway\"]
CMD [\"/opt/codegraph/bin/codegraph\", \"serve\", \"--mcp\"]
"""


RUNTIMES = {
    "gitnexus": ManagedMCPRuntimeSpec(
        "gitnexus",
        "gitnexus@1.6.3+fugue-vector.6",
        _gitnexus_runtime(),
        ("gitnexus", "mcp"),
        package_integrity=(
            "sha512-Yvhc70ESXFHPMtXHSddDgNL3dUxdvmA+"
            "CmxoadFBB7rTxIcXi8vrY/jhMvjvzJpBxp4JKi+lD8Pt1m1wL88Mtg=="
        ),
        asset_integrities=(
            (
                "Snowflake/snowflake-arctic-embed-xs",
                "git:d8c86521100d3556476a063fc2342036d45c106f",
            ),
            ("embedding_dimensions", "384"),
            ("onnxruntime-node", "version:1.24.3;cpu-only"),
            ("lexical_index", "flat-bm25-v1"),
            ("vector_index", "flat-cosine-v1;max-nodes:50000"),
        ),
        prepare_command=(
            "gitnexus",
            "analyze",
            "--skip-agents-md",
            "--force",
            "/workspace/repository",
        ),
        repository_state_paths=(".gitnexus",),
        runtime_env=(
            ("GITNEXUS_HOME", "/workspace/state/home/.gitnexus"),
            ("HF_HUB_OFFLINE", "1"),
            ("TRANSFORMERS_OFFLINE", "1"),
            (
                "FUGUE_GITNEXUS_MODEL_DIGEST",
                "cf2698d30ff05da02c70a088313bad56e5c2f401d734cb24a8390d446111936c",
            ),
        ),
        probe_command=(
            "node",
            "-e",
            "import('/opt/gitnexus-runtime/node_modules/gitnexus/dist/mcp/core/"
            "embedder.js').then(async m => { const v=await m.embedQuery('offline "
            "semantic readiness'); if(v.length!==384) process.exit(2); "
            "console.log(v.length) })",
        ),
    ),
    "codegraph": ManagedMCPRuntimeSpec(
        "codegraph",
        "@colbymchenry/codegraph@0.9.0",
        _codegraph_runtime(),
        ("/opt/codegraph/bin/codegraph", "serve", "--mcp"),
        package_integrity=(
            "sha512-pcqkItJ58hMZOJ+cNeENpllfopRfG47vTLyWWgKoPAfnZFeTz7wa69mf"
            "KtEVxUYO4RClXXJG4MdjN8SNqMRf2w=="
        ),
        platform_integrities=(
            (
                "amd64",
                "sha256:5f777f6005c6b6c75137cecd112911278e31f86c63f0cafaef367e74294eb629",
            ),
            (
                "arm64",
                "sha256:89e9eff1165d8251157640658d260aa0a95a8a6ee1da9c64f4cd2fa4136ba649",
            ),
        ),
        prepare_command=("/opt/codegraph/bin/codegraph", "init", "-i"),
        repository_state_paths=(".codegraph",),
    ),
    "semble": ManagedMCPRuntimeSpec(
        "semble",
        "semble[mcp]==0.5.1",
        _python_runtime(
            "'semble[mcp] @ https://files.pythonhosted.org/packages/d2/90/"
            "ad620429e2205a59f2eec0ace692e296b2ad328319e8480dcafd504d5e1e/"
            "semble-0.5.1-py3-none-any.whl#sha256="
            "dd31391ae284b9c29afb9562525e94d9294d175e6785bc26aeb36f1a46baf3bd' "
            "tree-sitter-language-pack==1.6.2",
            _SEMBLE_COMMAND,
            post_install=(
                '/opt/gateway/bin/python -c "from huggingface_hub import '
                "snapshot_download; snapshot_download(repo_id='minishlab/"
                "potion-code-16M-v2', revision='e9d2a44ca6a05ac6685f3b23709ea57e"
                "b7352d5b', local_dir='/opt/semble-model')\" && "
                '/opt/gateway/bin/python -c "import tree_sitter_language_pack '
                "as pack; languages='"
                + _SEMBLE_LANGUAGES
                + "'.split(','); pack.configure(cache_dir='/opt/tree-sitter-"
                "languages'); pack.download(languages); "
                '[pack.get_parser(name) for name in languages]"'
            ),
        ),
        _SEMBLE_COMMAND,
        package_integrity=(
            "sha256:dd31391ae284b9c29afb9562525e94d9294d175e6785bc26aeb36f1a46baf3bd"
        ),
        asset_integrities=(
            (
                "minishlab/potion-code-16M-v2",
                "git:e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b",
            ),
            (
                "tree-sitter-language-pack",
                "version:1.6.2;languages:" + _SEMBLE_LANGUAGES,
            ),
        ),
        runtime_env=(
            ("SEMBLE_MODEL_NAME", "/opt/semble-model"),
            ("SEMBLE_TREE_SITTER_CACHE", "/opt/tree-sitter-languages"),
            ("HF_HUB_OFFLINE", "1"),
        ),
        probe_command=(
            "/opt/gateway/bin/python",
            "-c",
            "import semble.mcp",
        ),
    ),
    "latmd": ManagedMCPRuntimeSpec(
        "latmd",
        "lat.md@0.11.0",
        _node_runtime("lat.md@0.11.0", ("lat", "mcp")),
        ("lat", "mcp"),
        package_integrity=(
            "sha512-MWq8OizSyw79FJkVbyjdlEvM9aivvdXbSmy92vsxTaJiIZ9twZuY/"
            "ZGGuydzuL4Xhu82z5bRz/8+ZOCowQ5dPQ=="
        ),
        prepare_command=("lat", "init"),
        repository_state_paths=("lat.md",),
    ),
    "project-rag": ManagedMCPRuntimeSpec(
        "project-rag",
        "git@d5abf98a48b60d35b73745e47e1aacca3963a6f0",
        f"""FROM {_PROJECT_RAG_RUST_IMAGE} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends clang cmake libclang-dev libprotobuf-dev pkg-config protobuf-compiler && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/Brainwires/project-rag /src && cd /src && git checkout --detach d5abf98a48b60d35b73745e47e1aacca3963a6f0 && test "$(git rev-parse HEAD)" = d5abf98a48b60d35b73745e47e1aacca3963a6f0 && test "$(git rev-parse HEAD^{{tree}})" = 16344729e08752575c278549cafbbd1cbd4e030d && cargo build --release
FROM {_PROJECT_RAG_PYTHON_IMAGE}
RUN python -m venv /opt/gateway && /opt/gateway/bin/pip install --no-cache-dir mcp==1.28.1 uvicorn==0.41.0 starlette==0.52.1
RUN /opt/gateway/bin/pip install --no-cache-dir huggingface_hub==1.23.0
COPY --from=builder /src/target/release/project-rag /usr/local/bin/project-rag
RUN HF_HOME=/opt/project-rag-model /opt/gateway/bin/python -c "from pathlib import Path; from huggingface_hub import snapshot_download; revision='5f1b8cd78bc4fb444dd171e59b18f3a3af89a079'; snapshot_download(repo_id='Qdrant/all-MiniLM-L6-v2-onnx', revision=revision); ref=Path('/opt/project-rag-model/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/refs/main'); ref.parent.mkdir(parents=True, exist_ok=True); ref.write_text(revision)"
RUN set -eu; mkdir -p /opt/project-rag-prewarm/data /opt/project-rag-prewarm/cache /opt/project-rag-prewarm/config; HOME=/opt/project-rag-prewarm XDG_DATA_HOME=/opt/project-rag-prewarm/data XDG_CACHE_HOME=/opt/project-rag-prewarm/cache XDG_CONFIG_HOME=/opt/project-rag-prewarm/config PROJECT_RAG_LANCEDB_PATH=/opt/project-rag-prewarm/lancedb HF_HOME=/opt/project-rag-model HF_HUB_OFFLINE=1 project-rag </dev/null >/tmp/project-rag-prewarm.log 2>&1 || true; grep -q "Using LanceDB vector database backend" /tmp/project-rag-prewarm.log; test -e "$(find /opt/project-rag-model -name model.onnx -print -quit)"; rm -rf /opt/project-rag-prewarm /tmp/project-rag-prewarm.log
COPY mcp_gateway.py /opt/fugue/mcp_gateway.py
COPY start-gateway /opt/fugue/start-gateway
RUN chmod 0555 /opt/fugue/start-gateway /usr/local/bin/project-rag
ENTRYPOINT [\"/opt/fugue/start-gateway\"]
CMD [\"project-rag\"]
""",
        ("project-rag",),
        package_integrity=("git-tree:16344729e08752575c278549cafbbd1cbd4e030d"),
        asset_integrities=(
            (
                "Qdrant/all-MiniLM-L6-v2-onnx",
                "git:5f1b8cd78bc4fb444dd171e59b18f3a3af89a079",
            ),
        ),
        runtime_env=(
            ("HF_HOME", "/opt/project-rag-model"),
            ("HF_HUB_OFFLINE", "1"),
            ("XDG_DATA_HOME", "/workspace/state/data"),
            ("XDG_CACHE_HOME", "/workspace/state/cache"),
            ("XDG_CONFIG_HOME", "/workspace/state/config"),
            ("PROJECT_RAG_LANCEDB_PATH", "/workspace/state/lancedb"),
            ("RUST_LOG", "off"),
        ),
    ),
}


def runtime_spec(system_id: str) -> ManagedMCPRuntimeSpec | None:
    return RUNTIMES.get(system_id)


def gitnexus_retrieval_mode(config: dict[str, Any] | None) -> str:
    selected = config or {}
    mode = str(selected.get("retrieval_mode") or "")
    if mode not in {"bm25", GITNEXUS_VECTOR_MODE}:
        raise ValueError("GitNexus retrieval_mode must be bm25 or hybrid_vector")
    if mode == GITNEXUS_VECTOR_MODE:
        expected = {
            "embedding_model": "Snowflake/snowflake-arctic-embed-xs",
            "embedding_revision": "d8c86521100d3556476a063fc2342036d45c106f",
            "embedding_dimensions": GITNEXUS_VECTOR_DIMENSIONS,
            "vector_required": True,
        }
        mismatched = [
            name for name, value in expected.items() if selected.get(name) != value
        ]
        if mismatched:
            raise ValueError(
                "GitNexus hybrid_vector requires the pinned vector contract: "
                + ", ".join(mismatched)
            )
    return mode


def prepare_runtime(
    system_id: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    spec = RUNTIMES.get(system_id)
    if spec is None:
        raise ValueError(f"context system has no managed MCP runtime: {system_id}")
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required to prepare managed MCP runtimes")
    root = repo_root / RUNTIME_ROOT / system_id
    build = root / "build"
    build.mkdir(parents=True, exist_ok=True)
    (build / "Dockerfile").write_text(spec.dockerfile)
    gateway = Path(__file__).resolve().parents[1] / "mcp_gateway.py"
    shutil.copy2(gateway, build / "mcp_gateway.py")
    (build / "start-gateway").write_text(_gateway_entrypoint())
    for asset in _runtime_build_assets(system_id):
        shutil.copy2(asset, build / asset.name)
    subprocess.run(
        [
            "docker",
            "build",
            "--provenance=false",
            "--pull",
            "-t",
            spec.image,
            build.as_posix(),
        ],
        cwd=repo_root,
        check=True,
        timeout=1800,
    )
    inspected = _inspect_image(spec.image)
    architecture = str(inspected.get("Architecture") or "")
    if architecture not in spec.architectures:
        raise RuntimeError(
            f"managed runtime {system_id} does not support {architecture or 'unknown'}"
        )
    lock = {
        "schema_version": 1,
        "system_id": system_id,
        "version": spec.version,
        "recipe_sha256": spec.recipe_sha256,
        "image": spec.image,
        "image_id": inspected["Id"],
        "repo_digests": sorted(inspected.get("RepoDigests") or []),
        "architecture": architecture,
        "os": inspected.get("Os"),
        "upstream_command": list(spec.upstream_command),
        "package_integrity": spec.package_integrity,
        "platform_integrities": dict(spec.platform_integrities),
        "asset_integrities": dict(spec.asset_integrities),
        "entrypoint": list(spec.entrypoint),
        "health_check": list(spec.health_check),
        "network_policy": spec.network_policy,
        "repository_mount": spec.repository_mount,
        "state_mount": spec.state_mount,
        "repository_state_paths": list(spec.repository_state_paths),
        "runtime_env": dict(spec.runtime_env),
    }
    atomic_write_json(root / "runtime-lock.json", lock)
    return lock


def read_runtime_lock(system_id: str, repo_root: Path) -> dict[str, Any] | None:
    spec = RUNTIMES.get(system_id)
    path = repo_root / RUNTIME_ROOT / system_id / "runtime-lock.json"
    if spec is None or not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("recipe_sha256") != spec.recipe_sha256:
        return None
    return value


def runtime_ready(system_id: str, repo_root: Path) -> tuple[bool, str]:
    spec = RUNTIMES.get(system_id)
    if spec is None:
        return False, "context system has no managed MCP runtime"
    lock = read_runtime_lock(system_id, repo_root)
    if lock is None:
        return False, "run fugue setup --prepare to build the pinned runtime"
    try:
        inspected = _inspect_image(str(lock["image"]))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return False, f"managed runtime image is unavailable: {exc}"
    if inspected.get("Id") != lock.get("image_id"):
        return False, "managed runtime image does not match runtime-lock.json"
    if inspected.get("Architecture") not in spec.architectures:
        return False, "managed runtime image architecture is not supported"
    return True, f"{lock['image']} matches {str(lock['image_id'])[:19]}"


def prepare_runtime_repository(
    system_id: str,
    *,
    repo_root: Path,
    artifact: Path,
    env: dict[str, str],
    config: dict[str, Any] | None = None,
    semantic_probe: Any = None,
) -> None:
    spec = RUNTIMES.get(system_id)
    if spec is None or not spec.prepare_command:
        return
    lock = read_runtime_lock(system_id, repo_root)
    if lock is None:
        raise RuntimeError(f"managed runtime for {system_id} is not prepared")
    repository = artifact / "repository"
    if not repository.is_dir():
        raise RuntimeError(f"prepared context {system_id} has no repository")
    home = artifact / "home"
    home.mkdir(parents=True, exist_ok=True)
    selected_env = [
        item
        for name in sorted(env)
        if name in {"LAT_LLM_KEY"} and env.get(name)
        for item in ("--env", name)
    ]
    docker_env = {
        **os.environ,
        **{name: value for name, value in env.items() if name in {"LAT_LLM_KEY"}},
    }
    mode = gitnexus_retrieval_mode(config) if system_id == "gitnexus" else None
    runtime_values = {
        **dict(spec.runtime_env),
        **(
            {"FUGUE_GITNEXUS_VECTOR_REQUIRED": "1"}
            if mode == GITNEXUS_VECTOR_MODE
            else {}
        ),
    }
    runtime_env = [
        item
        for name, value in runtime_values.items()
        for item in ("--env", f"{name}={value}")
    ]
    prepare_command = list(spec.prepare_command)
    _run_repository_prepare(
        repo_root=repo_root,
        repository=repository,
        home=home,
        spec=spec,
        image=str(lock["image_id"]),
        command=prepare_command,
        selected_env=selected_env,
        runtime_env=runtime_env,
        docker_env=docker_env,
    )
    if mode == GITNEXUS_VECTOR_MODE:
        nodes, _ = _gitnexus_graph_stats(artifact)
        if nodes > 50_000:
            raise RuntimeError(
                "GitNexus hybrid_vector is not applicable: graph has "
                f"{nodes} nodes; upstream vector safety limit is 50000"
            )
        vector_command = list(spec.prepare_command)
        vector_command.insert(2, "--embeddings")
        _run_repository_prepare(
            repo_root=repo_root,
            repository=repository,
            home=home,
            spec=spec,
            image=str(lock["image_id"]),
            command=vector_command,
            selected_env=selected_env,
            runtime_env=runtime_env,
            docker_env=docker_env,
        )
        _probe_gitnexus_index(
            repo_root,
            artifact,
            str(lock["image_id"]),
            runtime_values,
            semantic_probe=semantic_probe,
        )


def _run_repository_prepare(
    *,
    repo_root: Path,
    repository: Path,
    home: Path,
    spec: ManagedMCPRuntimeSpec,
    image: str,
    command: list[str],
    selected_env: list[str],
    runtime_env: list[str],
    docker_env: dict[str, str],
) -> None:
    invocation = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=1g",
        "--mount",
        f"type=bind,src={repository.resolve()},dst={spec.repository_mount}",
        "--mount",
        f"type=bind,src={home.resolve()},dst={spec.state_mount}/home",
        "--env",
        f"HOME={spec.state_mount}/home",
        *selected_env,
        *runtime_env,
        "--workdir",
        spec.repository_mount,
        "--entrypoint",
        command[0],
        image,
        *command[1:],
    ]
    try:
        subprocess.run(
            invocation,
            cwd=repo_root,
            env=docker_env,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.CalledProcessError as exc:
        detail = f"{exc.stdout or ''}\n{exc.stderr or ''}".strip()[-2_000:]
        raise RuntimeError(
            f"{spec.system_id} repository preparation failed: "
            f"{detail or f'exit code {exc.returncode}'}"
        ) from exc


def _gitnexus_graph_stats(artifact: Path) -> tuple[int, int]:
    metadata_path = artifact / "repository/.gitnexus/meta.json"
    if not metadata_path.is_file():
        raise RuntimeError("GitNexus setup did not create graph metadata")
    metadata = json.loads(metadata_path.read_text())
    stats = metadata.get("stats") if isinstance(metadata, dict) else None
    if not isinstance(stats, dict):
        raise RuntimeError("GitNexus setup produced invalid graph metadata")
    return int(stats.get("nodes") or 0), int(stats.get("embeddings") or 0)


def _probe_gitnexus_index(
    repo_root: Path,
    artifact: Path,
    image: str,
    runtime_env: dict[str, str],
    *,
    semantic_probe: Any = None,
) -> None:
    nodes, embeddings = _gitnexus_graph_stats(artifact)
    if nodes > 50_000:
        raise RuntimeError(
            "GitNexus hybrid_vector is not applicable: graph has "
            f"{nodes} nodes; upstream vector safety limit is 50000"
        )
    if embeddings <= 0:
        raise RuntimeError("GitNexus hybrid setup produced zero embeddings")
    query = "semantic implementation relationship"
    expected_path = None
    if semantic_probe is not None:
        if not isinstance(semantic_probe, dict):
            raise ValueError("GitNexus semantic_probe must be a mapping")
        query = str(semantic_probe.get("query") or "").strip()
        expected_path = str(semantic_probe.get("expected_path") or "").strip()
        if not query or not expected_path or expected_path.startswith("/"):
            raise ValueError(
                "GitNexus semantic_probe requires query and relative expected_path"
            )
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        (
            f"type=bind,src={(artifact / 'repository').resolve()},"
            "dst=/workspace/repository"
        ),
        "--mount",
        f"type=bind,src={(artifact / 'home').resolve()},dst=/workspace/state/home",
        "--env",
        "HOME=/workspace/state/home",
    ]
    for name, value in runtime_env.items():
        command.extend(("--env", f"{name}={value}"))
    command.extend(
        (
            "--workdir",
            "/workspace/repository",
            "--entrypoint",
            "gitnexus",
            image,
            "query",
            query,
            "--limit",
            "1",
        )
    )
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    evidence = f"{result.stdout}\n{result.stderr}"
    if result.returncode:
        raise RuntimeError(
            "GitNexus offline semantic query failed: " + evidence.strip()[-2_000:]
        )
    if '"vector_search_succeeded":true' not in evidence:
        raise RuntimeError(
            "GitNexus offline semantic query did not report successful vector search"
        )
    if expected_path and expected_path not in result.stdout:
        raise RuntimeError(
            "GitNexus semantic contract did not retrieve expected path: "
            + expected_path
        )


def query_gitnexus(
    *,
    repo_root: Path,
    artifact: Path,
    config: dict[str, Any],
    query: str,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = gitnexus_retrieval_mode(config)
    lock = read_runtime_lock("gitnexus", repo_root)
    if lock is None:
        raise RuntimeError("managed runtime for gitnexus is not prepared")
    runtime_values = {
        **dict(RUNTIMES["gitnexus"].runtime_env),
        **(
            {"FUGUE_GITNEXUS_VECTOR_REQUIRED": "1"}
            if mode == GITNEXUS_VECTOR_MODE
            else {}
        ),
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        f"type=bind,src={(artifact / 'repository').resolve()},dst=/workspace/repository",
        "--mount",
        f"type=bind,src={(artifact / 'home').resolve()},dst=/workspace/state/home",
        "--env",
        "HOME=/workspace/state/home",
    ]
    for name, value in runtime_values.items():
        command.extend(("--env", f"{name}={value}"))
    command.extend(
        (
            "--workdir",
            "/workspace/repository",
            "--entrypoint",
            "gitnexus",
            str(lock["image_id"]),
            "query",
            query,
            "--limit",
            str(top_k),
        )
    )
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode:
        detail = f"{result.stdout}\n{result.stderr}".strip()[-2_000:]
        raise RuntimeError(f"GitNexus offline query failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GitNexus offline query returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("GitNexus offline query returned a non-object result")
    telemetry: dict[str, Any] = {"retrieval_mode": mode}
    for line in result.stderr.splitlines():
        marker = "FUGUE_GITNEXUS_VECTOR "
        if marker not in line:
            continue
        try:
            value = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            telemetry.update(value)
    if "model_digest" in telemetry:
        telemetry["vector_model_digest"] = telemetry.pop("model_digest")
    if "query_latency_ms" in telemetry:
        telemetry["vector_query_latency_ms"] = telemetry.pop("query_latency_ms")
    if (
        mode == GITNEXUS_VECTOR_MODE
        and telemetry.get("vector_search_succeeded") is not True
    ):
        raise RuntimeError("GitNexus hybrid query did not execute vector retrieval")
    return payload, telemetry


def run_runtime_command(
    system_id: str,
    *,
    repo_root: Path,
    repository: Path,
    env: dict[str, str],
    command: tuple[str, ...],
    timeout: float = 300,
) -> subprocess.CompletedProcess[str]:
    spec = RUNTIMES.get(system_id)
    lock = read_runtime_lock(system_id, repo_root)
    if spec is None or lock is None:
        raise RuntimeError(f"managed runtime for {system_id} is not prepared")
    if not command or command[0] != spec.upstream_command[0]:
        raise ValueError(f"managed runtime {system_id} rejected an unknown command")
    selected_env = [
        item for name in ("LAT_LLM_KEY",) if env.get(name) for item in ("--env", name)
    ]
    docker_env = {
        **os.environ,
        **{name: value for name, value in env.items() if name in {"LAT_LLM_KEY"}},
    }
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--tmpfs",
            f"{spec.state_mount}:rw,noexec,nosuid,size=256m",
            "--mount",
            f"type=bind,src={repository.resolve()},dst={spec.repository_mount}",
            "--env",
            f"HOME={spec.state_mount}/home",
            *selected_env,
            "--workdir",
            spec.repository_mount,
            "--entrypoint",
            command[0],
            str(lock["image_id"]),
            *command[1:],
        ],
        cwd=repo_root,
        env=docker_env,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def render_runtime_compose(
    system_id: str,
    *,
    repo_root: Path,
    artifact: Path,
    runtime_root: Path,
    job_name: str,
    env_names: tuple[str, ...],
    write: bool,
    context_config: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    spec = RUNTIMES[system_id]
    lock = read_runtime_lock(system_id, repo_root)
    if lock is None:
        raise RuntimeError(f"managed runtime for {system_id} is not prepared")
    service = f"fugue-{system_id}"
    mode = gitnexus_retrieval_mode(context_config) if system_id == "gitnexus" else None
    context_env = (
        {"FUGUE_GITNEXUS_VECTOR_REQUIRED": "1"} if mode == GITNEXUS_VECTOR_MODE else {}
    )
    evidence_dir = runtime_root / "gateway-evidence" / job_name
    event_log = evidence_dir / "context-gateway.jsonl"
    compose = {
        "services": {
            service: {
                "image": lock["image_id"],
                "pull_policy": "never",
                "network_mode": "service:main",
                "read_only": True,
                "security_opt": ["no-new-privileges:true"],
                "cap_drop": ["ALL"],
                "volumes": [
                    f"{artifact.resolve().as_posix()}:/fugue-context:ro",
                    (
                        f"{(artifact / 'repository').resolve().as_posix()}:"
                        f"{spec.repository_mount}:ro"
                    ),
                    {
                        "type": "bind",
                        "source": evidence_dir.resolve().as_posix(),
                        "target": "/fugue-evidence",
                        "read_only": False,
                        "bind": {"create_host_path": False},
                    },
                ],
                "tmpfs": [
                    "/tmp:rw,noexec,nosuid,size=64m",
                    f"{spec.state_mount}:rw,noexec,nosuid,size=2g",
                    *[
                        f"{spec.repository_mount}/{path}:rw,noexec,nosuid,size=2g"
                        for path in spec.repository_state_paths
                    ],
                ],
                "environment": {
                    "HOME": "/workspace/state/home",
                    **dict(spec.runtime_env),
                    **context_env,
                    "FUGUE_REPOSITORY_STATE_PATHS": " ".join(
                        spec.repository_state_paths
                    ),
                    "FUGUE_CONTEXT_SYSTEM_ID": system_id,
                    "FUGUE_GATEWAY_EVENT_LOG": (
                        "/fugue-evidence/context-gateway.jsonl"
                    ),
                    **{
                        name: f"${{{name}}}"
                        for name in (
                            "FUGUE_RUN_ID",
                            "FUGUE_CANDIDATE_ID",
                            "FUGUE_COMPARISON_EXAMPLE_ID",
                            "FUGUE_TRIAL_INDEX",
                            "FUGUE_EXECUTION_FINGERPRINT",
                            "FUGUE_WEAVE_CONVERSATION_ID",
                        )
                    },
                    **{name: f"${{{name}}}" for name in env_names},
                },
                "healthcheck": {
                    "test": [
                        "CMD",
                        *spec.health_check,
                    ],
                    "interval": "5s",
                    "timeout": "3s",
                    "retries": 30,
                    "start_period": "10s",
                },
            }
        }
    }
    path = runtime_root / "context-runtimes" / f"{job_name}.yaml"
    if write:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        event_log.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(compose, sort_keys=False))
    server = {
        "name": system_id,
        "transport": "streamable-http",
        "url": f"http://127.0.0.1:{GATEWAY_PORT}/mcp",
    }
    descriptor = {
        "schema_version": 1,
        "system_id": system_id,
        "recipe_sha256": lock["recipe_sha256"],
        "image": lock["image"],
        "image_id": lock["image_id"],
        "retrieval_mode": mode,
        "vector_required": mode == GITNEXUS_VECTOR_MODE,
    }
    return path, server, descriptor


def _gateway_entrypoint() -> str:
    return """#!/bin/sh
set -eu
mkdir -p /workspace/state/home
if [ -d /fugue-context/home ]; then cp -a /fugue-context/home/. /workspace/state/home/; fi
for relative in ${FUGUE_REPOSITORY_STATE_PATHS:-}; do
  source="/fugue-context/repository/$relative"
  target="/workspace/repository/$relative"
  if [ -d "$source" ]; then cp -a "$source"/. "$target"/; fi
done
cd /workspace/repository
exec /opt/gateway/bin/python /opt/fugue/mcp_gateway.py --host 0.0.0.0 --port 8765 -- "$@"
"""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
