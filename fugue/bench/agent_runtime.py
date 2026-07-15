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

from filelock import FileLock

AGENT_RUNTIME_ROOT = Path(".fugue/runtime/agent-runtimes")
AGENT_RUNTIME_MOUNT = "/opt/fugue-agent-runtime"
_NODE_IMAGE = (
    "node:22.23.0-bookworm-slim@"
    "sha256:d9f850096136edbc402debdd8729579a288aac64574ada0ff4db26b6ae58b0b2"
)
_PYTHON_IMAGE = (
    "python:3.12.11-slim-bookworm@"
    "sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
)
_RUST_IMAGE = (
    "rust:1.95.0-bookworm@"
    "sha256:6258907abe69656e41cd992e0b705cdcfabcbbe3db374f92ed2d47121282d4a1"
)
_CODEX_SOURCE_COMMIT = "c4d748f586a84a3ed5b6aceb82e9a1db4abb1cda"
_CODEX_SOURCE_SHA256 = (
    "bd8e3859e8d30837624562496c1ef253ecf36785edf7ef2b78b938be83a2ab02"
)
_CODEX_MCP_TARGET_SHA256 = (
    "4f09adb59d617f062f0db40cdfbd8b8aad0794e1716d2f9042dc70bf264c3c97"
)
_CODEX_CARGO_LOCK_SHA256 = (
    "e3a9e6d6d98c1c4bd5e06cd8c8df16e239c8cbdc0b42d6e5ae10be734d08a9d3"
)
_CODEX_RELEASE_CARGO_LOCK_SHA256 = (
    "3b07f49540302b87e84adf853a8c2a23325228d14be25cebe7ca422d66953e77"
)


@dataclass(frozen=True)
class AgentRuntimeSpec:
    harness: str
    version: str
    dockerfile: str
    probe: tuple[str, ...]
    platform: str = "linux/amd64"
    architectures: tuple[str, ...] = ("amd64", "arm64")

    @property
    def recipe_sha256(self) -> str:
        return _digest(
            {
                "spec": asdict(self),
                "assets": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in _build_assets(self.harness)
                },
            }
        )

    @property
    def image(self) -> str:
        return self.image_for(self.platform.rsplit("/", 1)[-1])

    def image_for(self, architecture: str) -> str:
        return f"fugue-agent-{self.harness}-{architecture}:{self.recipe_sha256[:12]}"


def _codex_dockerfile() -> str:
    source_root = f"codex-{_CODEX_SOURCE_COMMIT}"
    return f"""FROM {_RUST_IMAGE} AS codex-builder
ARG CODEX_SOURCE_COMMIT={_CODEX_SOURCE_COMMIT}
ARG CODEX_SOURCE_SHA256={_CODEX_SOURCE_SHA256}
ARG CODEX_MCP_TARGET_SHA256={_CODEX_MCP_TARGET_SHA256}
ARG CODEX_CARGO_LOCK_SHA256={_CODEX_CARGO_LOCK_SHA256}
ARG CODEX_RELEASE_CARGO_LOCK_SHA256={_CODEX_RELEASE_CARGO_LOCK_SHA256}
WORKDIR /src
RUN curl -fsSLo codex.tar.gz \
      "https://github.com/openai/codex/archive/${{CODEX_SOURCE_COMMIT}}.tar.gz" && \
    echo "${{CODEX_SOURCE_SHA256}}  codex.tar.gz" | sha256sum -c - && \
    tar -xzf codex.tar.gz && rm codex.tar.gz
COPY codex-flat-mcp.patch /tmp/codex-flat-mcp.patch
WORKDIR /src/{source_root}
RUN echo "${{CODEX_MCP_TARGET_SHA256}}  codex-rs/core/src/tools/handlers/mcp.rs" \
      | sha256sum -c - && \
    patch -p1 --fuzz=0 < /tmp/codex-flat-mcp.patch && \
    echo "${{CODEX_CARGO_LOCK_SHA256}}  codex-rs/Cargo.lock" \
      | sha256sum -c - && \
    sed -i 's/^version = "0\\.0\\.0"$/version = "0.143.0"/' \
      codex-rs/Cargo.lock && \
    echo "${{CODEX_RELEASE_CARGO_LOCK_SHA256}}  codex-rs/Cargo.lock" \
      | sha256sum -c -
WORKDIR /src/{source_root}/codex-rs
RUN cargo build --locked --release -p codex-cli
RUN mkdir -p /opt/codex-libs && \
    cp /usr/lib/*-linux-gnu/libssl.so.3 /opt/codex-libs/ && \
    cp /usr/lib/*-linux-gnu/libcrypto.so.3 /opt/codex-libs/

FROM {_NODE_IMAGE}
WORKDIR {AGENT_RUNTIME_MOUNT}
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY patch-runtime.mjs ./
RUN node patch-runtime.mjs node_modules/weave-codex
RUN mkdir -p bin lib libexec && \
    ln -s ../node_modules/.bin/weave-codex bin/weave-codex && \
    cp /usr/local/bin/node bin/node && \
    rm -f node_modules/.bin/codex
COPY --from=codex-builder \
  /src/{source_root}/codex-rs/target/release/codex libexec/codex
COPY --from=codex-builder /opt/codex-libs/ lib/
COPY codex-wrapper.sh bin/codex
RUN chmod 0755 bin/codex libexec/codex && \
    PATH={AGENT_RUNTIME_MOUNT}/bin:$PATH codex --version && \
    PATH={AGENT_RUNTIME_MOUNT}/bin:$PATH weave-codex --help >/dev/null
"""


def _node_agent_dockerfile(harness: str, command: str, version: str) -> str:
    patch_lock = f"{harness}-patch-lock.json"
    postinstall = (
        ""
        if harness == "openclaw"
        else "RUN node node_modules/@anthropic-ai/claude-code/install.cjs\n"
    )
    weave_cli = (
        ""
        if harness == "openclaw"
        else (
            " && ln -s ../node_modules/.bin/weave-claude-code "
            "bin/weave-claude-code && "
            "ln -s ../lib/node_modules/npm/bin/npm-cli.js bin/npm"
        )
    )
    weave_global_view = (
        ""
        if harness == "openclaw"
        else (
            "RUN mkdir -p lib/node_modules && "
            "cp -a \"$(npm root -g)/npm\" lib/node_modules/npm && "
            "ln -s ../../node_modules/weave-claude-code "
            "lib/node_modules/weave-claude-code && "
            f"export NPM_CONFIG_PREFIX={AGENT_RUNTIME_MOUNT} && "
            "test -s \"$(npm root -g)/weave-claude-code/"
            ".claude-plugin/marketplace.json\"\n"
        )
    )
    return f"""FROM {_NODE_IMAGE}
WORKDIR {AGENT_RUNTIME_MOUNT}
COPY package.json package-lock.json ./
{("COPY weave-node-sdk.tgz ./" if harness == "openclaw" else "")}
RUN npm ci --ignore-scripts --no-audit --no-fund
{postinstall}{weave_global_view}COPY patch-runtime.mjs ./
RUN node patch-runtime.mjs {AGENT_RUNTIME_MOUNT} && test -s {patch_lock}
RUN mkdir -p bin && cp /usr/local/bin/node bin/node && \
    ln -s ../node_modules/.bin/{command} bin/{command}{weave_cli}
RUN PATH={AGENT_RUNTIME_MOUNT}/bin:$PATH {command} --version | grep -F {json.dumps(version)}
"""


def _hermes_dockerfile() -> str:
    plugin_commit = "670e98ff503e574994e6e64fa0d1294f4d7eefdf"
    return f"""FROM {_PYTHON_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git ripgrep xz-utils && \
    rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 fugue && \
    mkdir -p {AGENT_RUNTIME_MOUNT} && chown -R fugue:fugue {AGENT_RUNTIME_MOUNT}
ENV HOME={AGENT_RUNTIME_MOUNT}/home
WORKDIR {AGENT_RUNTIME_MOUNT}
USER fugue
RUN curl -fsSLo /tmp/hermes-install.sh \
      https://raw.githubusercontent.com/NousResearch/hermes-agent/v2026.6.5/scripts/install.sh && \
    echo "5562d544934751313f16c57ed3dd17b052600cd8fae9f6e6977bfc9ef1e72f38  /tmp/hermes-install.sh" | sha256sum -c - && \
    bash /tmp/hermes-install.sh --skip-setup --skip-browser --no-skills \
      --non-interactive --branch v2026.6.5 && rm /tmp/hermes-install.sh
RUN curl -fsSLo /tmp/hermes-otel.tar.gz \
      https://github.com/briancaffey/hermes-otel/archive/{plugin_commit}.tar.gz && \
    echo "6c3cbe6a9e3d65fa79714591ef1a2e7a3c58df4c90889923acbd069ed7a3f054  /tmp/hermes-otel.tar.gz" | sha256sum -c - && \
    mkdir hermes-otel && tar -xzf /tmp/hermes-otel.tar.gz -C hermes-otel --strip-components=1 && \
    rm /tmp/hermes-otel.tar.gz
COPY patch-plugin.py ./
RUN python patch-plugin.py && \
    VENV_PY="$HOME/.hermes/hermes-agent/venv/bin/python" && \
    test -x "$VENV_PY" && \
    "$HOME/.hermes/bin/uv" pip install --quiet --python "$VENV_PY" -e \
      "{AGENT_RUNTIME_MOUNT}/hermes-otel[yaml]"
RUN mkdir -p bin && ln -s ../home/.local/bin/hermes bin/hermes && \
    PATH={AGENT_RUNTIME_MOUNT}/bin:$PATH hermes version
"""


RUNTIMES = {
    "hermes": AgentRuntimeSpec(
        harness="hermes",
        version="hermes-agent@v2026.6.5+hermes-otel@670e98f+fugue-span-attrs.2",
        dockerfile=_hermes_dockerfile(),
        probe=("/bin/sh", "-c", f"PATH={AGENT_RUNTIME_MOUNT}/bin:$PATH hermes version"),
    ),
    "openclaw": AgentRuntimeSpec(
        harness="openclaw",
        version=(
            "openclaw@2026.7.1+weave-openclaw@0.1.1+"
            "weave-otel2.1+fugue-load-path.1"
        ),
        dockerfile=_node_agent_dockerfile("openclaw", "openclaw", "2026.7.1"),
        probe=(
            "/bin/sh",
            "-c",
            f"PATH={AGENT_RUNTIME_MOUNT}/bin:$PATH openclaw --version && "
            f"test -s {AGENT_RUNTIME_MOUNT}/node_modules/weave-openclaw/"
            "openclaw.plugin.json",
        ),
    ),
    "claude-code": AgentRuntimeSpec(
        harness="claude-code",
        version="claude-code@2.1.210+weave-claude-code@0.2.12+fugue-attrs.3",
        dockerfile=_node_agent_dockerfile("claude-code", "claude", "2.1.210"),
        probe=(
            "/bin/sh",
            "-c",
            f"export PATH={AGENT_RUNTIME_MOUNT}/bin:$PATH && "
            "claude --version && "
            f"export NPM_CONFIG_PREFIX={AGENT_RUNTIME_MOUNT} && "
            "npm root -g | "
            f"grep -F {AGENT_RUNTIME_MOUNT}/lib/node_modules && "
            "test -s \"$(npm root -g)/weave-claude-code/"
            ".claude-plugin/marketplace.json\"",
        ),
    ),
    "codex": AgentRuntimeSpec(
        harness="codex",
        version=("codex@0.143.0+fugue-flat-mcp.1+weave-codex@0.1.1+fugue-mcp-meta.1"),
        dockerfile=_codex_dockerfile(),
        probe=(
            "/bin/sh",
            "-c",
            f"PATH={AGENT_RUNTIME_MOUNT}/bin:$PATH codex --version && "
            "weave-codex --help >/dev/null",
        ),
    ),
}


def runtime_spec(harness: str) -> AgentRuntimeSpec | None:
    return RUNTIMES.get(harness)


def prepare_runtime(
    harness: str,
    *,
    repo_root: Path,
    architecture: str = "amd64",
    rebuild: bool = False,
) -> dict[str, Any]:
    spec = RUNTIMES.get(harness)
    if spec is None:
        raise ValueError(f"harness has no prepared agent runtime: {harness}")
    if architecture not in spec.architectures:
        raise ValueError(
            f"agent runtime {harness} does not support architecture {architecture}"
        )
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required to prepare agent runtimes")
    root = repo_root / AGENT_RUNTIME_ROOT / harness
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / ".prepare.lock", timeout=1800):
        existing = read_runtime_lock(harness, repo_root, architecture)
        if not rebuild and existing is not None:
            ready, _ = runtime_ready(harness, repo_root, architecture)
            if ready:
                return existing
        build = root / f"build-{uuid.uuid4().hex}"
        build.mkdir()
        try:
            (build / "Dockerfile").write_text(spec.dockerfile)
            for asset in _build_assets(harness):
                shutil.copy2(asset, build / asset.name)
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--provenance=false",
                    "--platform",
                    f"linux/{architecture}",
                    "--pull",
                    "-t",
                    spec.image_for(architecture),
                    build.as_posix(),
                ],
                cwd=repo_root,
                check=True,
                timeout=1800,
            )
            inspected = _inspect_image(spec.image_for(architecture))
            observed_architecture = str(inspected.get("Architecture") or "")
            if observed_architecture != architecture:
                raise RuntimeError(
                    f"agent runtime {harness} does not support "
                    f"{observed_architecture or 'unknown'}"
                )
            lock = {
                "schema_version": 1,
                "harness": harness,
                "version": spec.version,
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image_for(architecture),
                "image_id": inspected["Id"],
                "architecture": architecture,
                "os": inspected.get("Os"),
                "probe": list(spec.probe),
            }
            _atomic_json(root / f"runtime-lock-{architecture}.json", lock)
            return lock
        finally:
            shutil.rmtree(build, ignore_errors=True)


def read_runtime_lock(
    harness: str,
    repo_root: Path,
    architecture: str = "amd64",
) -> dict[str, Any] | None:
    spec = RUNTIMES.get(harness)
    path = (
        repo_root / AGENT_RUNTIME_ROOT / harness / f"runtime-lock-{architecture}.json"
    )
    legacy = repo_root / AGENT_RUNTIME_ROOT / harness / "runtime-lock.json"
    if architecture == "amd64" and not path.is_file() and legacy.is_file():
        path = legacy
    if spec is None or not path.is_file():
        return None
    value = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "harness": harness,
        "version": spec.version,
        "recipe_sha256": spec.recipe_sha256,
        "image": spec.image_for(architecture),
        "architecture": architecture,
    }
    if not isinstance(value, dict) or any(
        value.get(k) != v for k, v in required.items()
    ):
        return None
    return value


def runtime_ready(
    harness: str,
    repo_root: Path,
    architecture: str = "amd64",
) -> tuple[bool, str]:
    lock = read_runtime_lock(harness, repo_root, architecture)
    if lock is None:
        return False, "run fugue setup --prepare to build the pinned agent runtime"
    try:
        inspected = _inspect_image(str(lock["image_id"]))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return False, f"prepared agent runtime is unavailable: {exc}"
    if inspected.get("Id") != lock.get("image_id"):
        return False, "prepared agent runtime image drifted from its lock"
    return True, f"{lock['image']} matches {str(lock['image_id'])[:19]}"


def runtime_mount(
    harness: str,
    repo_root: Path,
    architecture: str = "amd64",
) -> dict[str, Any] | None:
    lock = read_runtime_lock(harness, repo_root, architecture)
    if lock is None:
        return None
    return {
        "type": "image",
        "source": str(lock["image_id"]),
        "target": AGENT_RUNTIME_MOUNT,
        "read_only": True,
        "image": {"subpath": AGENT_RUNTIME_MOUNT.lstrip("/")},
    }


def _build_assets(harness: str) -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2] / "configs/fugue/runtime" / harness
    assets = [path for path in sorted(root.iterdir()) if path.is_file()]
    if harness == "openclaw":
        assets.append(Path(__file__).resolve().parents[2] / "vendor/weave-node-sdk.tgz")
    return tuple(assets)


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
    if not isinstance(values, list) or len(values) != 1:
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
