from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

RUNTIME_ROOT = Path(".fugue/runtime/context-runtimes")
GATEWAY_PORT = 8765


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
    runtime_env: tuple[tuple[str, str], ...] = ()
    probe_command: tuple[str, ...] = ()

    @property
    def recipe_sha256(self) -> str:
        gateway = Path(__file__).resolve().parents[1] / "mcp_gateway.py"
        return _digest(
            {
                "spec": asdict(self),
                "gateway_sha256": hashlib.sha256(gateway.read_bytes()).hexdigest(),
                "entrypoint_sha256": hashlib.sha256(
                    _gateway_entrypoint().encode()
                ).hexdigest(),
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
        "gitnexus@1.6.3",
        _node_runtime(
            "gitnexus@1.6.3",
            ("gitnexus", "mcp"),
            ignore_scripts=False,
        ),
        ("gitnexus", "mcp"),
        package_integrity=(
            "sha512-Yvhc70ESXFHPMtXHSddDgNL3dUxdvmA+"
            "CmxoadFBB7rTxIcXi8vrY/jhMvjvzJpBxp4JKi+lD8Pt1m1wL88Mtg=="
        ),
        prepare_command=(
            "gitnexus",
            "analyze",
            "--skip-agents-md",
            "--force",
            "/workspace/repository",
        ),
        runtime_env=(("GITNEXUS_HOME", "/workspace/state/home/.gitnexus"),),
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
                "/opt/gateway/bin/python -c \"from huggingface_hub import "
                "snapshot_download; snapshot_download(repo_id='minishlab/"
                "potion-code-16M-v2', revision='e9d2a44ca6a05ac6685f3b23709ea57e"
                "b7352d5b', local_dir='/opt/semble-model')\" && "
                "/opt/gateway/bin/python -c \"import tree_sitter_language_pack "
                "as pack; languages='"
                + _SEMBLE_LANGUAGES
                + "'.split(','); pack.configure(cache_dir='/opt/tree-sitter-"
                "languages'); pack.download(languages); "
                "[pack.get_parser(name) for name in languages]\""
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
    ),
    "project-rag": ManagedMCPRuntimeSpec(
        "project-rag",
        "git@d5abf98a48b60d35b73745e47e1aacca3963a6f0",
        f"""FROM {_PROJECT_RAG_RUST_IMAGE} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends clang cmake libclang-dev libprotobuf-dev pkg-config protobuf-compiler && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/Brainwires/project-rag /src && cd /src && git checkout --detach d5abf98a48b60d35b73745e47e1aacca3963a6f0 && test "$(git rev-parse HEAD)" = d5abf98a48b60d35b73745e47e1aacca3963a6f0 && test "$(git rev-parse HEAD^{{tree}})" = 16344729e08752575c278549cafbbd1cbd4e030d && cargo build --release
FROM {_PROJECT_RAG_PYTHON_IMAGE}
RUN python -m venv /opt/gateway && /opt/gateway/bin/pip install --no-cache-dir mcp==1.28.1 uvicorn==0.41.0 starlette==0.52.1
COPY --from=builder /src/target/release/project-rag /usr/local/bin/project-rag
COPY mcp_gateway.py /opt/fugue/mcp_gateway.py
COPY start-gateway /opt/fugue/start-gateway
RUN chmod 0555 /opt/fugue/start-gateway /usr/local/bin/project-rag
ENTRYPOINT [\"/opt/fugue/start-gateway\"]
CMD [\"project-rag\"]
""",
        ("project-rag",),
        package_integrity=("git-tree:16344729e08752575c278549cafbbd1cbd4e030d"),
    ),
}


def runtime_spec(system_id: str) -> ManagedMCPRuntimeSpec | None:
    return RUNTIMES.get(system_id)


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
    subprocess.run(
        ["docker", "build", "--pull", "-t", spec.image, build.as_posix()],
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
        "runtime_env": dict(spec.runtime_env),
    }
    _atomic_json(root / "runtime-lock.json", lock)
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
        return False, "run fugue setup --prepare-context to build the pinned runtime"
    try:
        inspected = _inspect_image(str(lock["image"]))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return False, f"managed runtime image is unavailable: {exc}"
    if inspected.get("Id") != lock.get("image_id"):
        return False, "managed runtime image does not match runtime-lock.json"
    if inspected.get("Architecture") not in spec.architectures:
        return False, "managed runtime image architecture is not supported"
    return True, f"{lock['image']} matches {str(lock['image_id'])[:19]}"


def probe_runtime_install(system_id: str, repo_root: Path) -> None:
    spec = RUNTIMES.get(system_id)
    lock = read_runtime_lock(system_id, repo_root)
    if spec is None or lock is None:
        raise RuntimeError(f"managed runtime for {system_id} is not prepared")
    command = spec.install_probe_command
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--entrypoint",
            command[0],
            str(lock["image"]),
            *command[1:],
        ],
        cwd=repo_root,
        check=True,
        timeout=30,
    )


def prepare_runtime_repository(
    system_id: str,
    *,
    repo_root: Path,
    artifact: Path,
    env: dict[str, str],
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
    runtime_env = [
        item for name, value in spec.runtime_env for item in ("--env", f"{name}={value}")
    ]
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
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
            spec.prepare_command[0],
            str(lock["image"]),
            *spec.prepare_command[1:],
        ],
        cwd=repo_root,
        env=docker_env,
        check=True,
        timeout=1800,
    )


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
        item
        for name in ("LAT_LLM_KEY",)
        if env.get(name)
        for item in ("--env", name)
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
            str(lock["image"]),
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
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    spec = RUNTIMES[system_id]
    lock = read_runtime_lock(system_id, repo_root)
    if lock is None:
        raise RuntimeError(f"managed runtime for {system_id} is not prepared")
    service = f"fugue-{system_id}"
    compose = {
        "services": {
            service: {
                "image": lock["image"],
                "network_mode": "service:main",
                "read_only": True,
                "security_opt": ["no-new-privileges:true"],
                "cap_drop": ["ALL"],
                "volumes": [f"{artifact.resolve().as_posix()}:/fugue-context:ro"],
                "tmpfs": [
                    f"{spec.state_mount}:rw,noexec,nosuid,size=2g",
                    f"{spec.repository_mount}:rw,noexec,nosuid,size=2g",
                ],
                "environment": {
                    "HOME": "/workspace/state/home",
                    **dict(spec.runtime_env),
                    "FUGUE_CONTEXT_SYSTEM_ID": system_id,
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
    }
    return path, server, descriptor


def _gateway_entrypoint() -> str:
    return """#!/bin/sh
set -eu
mkdir -p /workspace/repository /workspace/state/home
cp -a /fugue-context/repository/. /workspace/repository/
if [ -d /fugue-context/home ]; then cp -a /fugue-context/home/. /workspace/state/home/; fi
cd /workspace/repository
exec /opt/gateway/bin/python /opt/fugue/mcp_gateway.py --host 0.0.0.0 --port 8765 -- "$@"
"""


def _inspect_image(image: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "image missing").strip())
    values = json.loads(result.stdout)
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise RuntimeError("docker image inspect returned invalid JSON")
    return values[0]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
