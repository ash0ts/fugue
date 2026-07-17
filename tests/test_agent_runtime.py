from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from fugue.bench import agent_runtime


def test_codex_runtime_is_locked_and_trial_install_is_verification_only() -> None:
    spec = agent_runtime.RUNTIMES["codex"]
    assert spec.version == (
        "codex@0.143.0+fugue-flat-mcp.1+weave-codex@0.1.1+fugue-mcp-meta.1+skill-use.1"
    )
    assert "npm ci --ignore-scripts" in spec.dockerfile
    assert "cargo build --locked --release -p codex-cli" in spec.dockerfile
    assert "patch -p1 --fuzz=0" in spec.dockerfile
    assert agent_runtime._CODEX_SOURCE_COMMIT in spec.dockerfile
    assert agent_runtime._CODEX_SOURCE_SHA256 in spec.dockerfile
    assert agent_runtime._CODEX_MCP_TARGET_SHA256 in spec.dockerfile
    assert agent_runtime._CODEX_CARGO_LOCK_SHA256 in spec.dockerfile
    assert agent_runtime._CODEX_RELEASE_CARGO_LOCK_SHA256 in spec.dockerfile
    assert "patch-runtime.mjs" in spec.dockerfile
    assert "libssl.so.3" in spec.dockerfile
    assert "libcrypto.so.3" in spec.dockerfile
    assert "codex-wrapper.sh" in spec.dockerfile
    source = (Path("fugue/agents/model_plane.py")).read_text()
    codex_install = source[
        source.index("class FugueCodex") :
    ]
    assert "npm install" not in codex_install
    assert "apt-get" not in codex_install
    assert "curl " not in codex_install
    assert "run fugue setup --prepare" in codex_install


def test_all_release_harnesses_are_setup_built_and_trial_verified() -> None:
    source = Path("fugue/agents/model_plane.py").read_text()
    ranges = {
        "hermes": source[
            source.index("class FugueHermes") : source.index("class FugueOpenClaw")
        ],
        "openclaw": source[
            source.index("class FugueOpenClaw") : source.index("class FugueClaudeCode")
        ],
        "claude-code": source[
            source.index("class FugueClaudeCode") : source.index("class FugueCodex")
        ],
        "codex": source[
            source.index("class FugueCodex") :
        ],
    }
    assert set(ranges) == set(agent_runtime.RUNTIMES)
    for harness, adapter in ranges.items():
        assert "run fugue setup --prepare" in adapter, harness
        for forbidden in ("apt-get", "npm install", "pip install", "curl -fs"):
            assert forbidden not in adapter, (harness, forbidden)

    hermes_adapter = ranges["hermes"]
    assert agent_runtime.RUNTIMES["hermes"].version.endswith("+single-agent.1")
    hermes_runtime = agent_runtime.RUNTIMES["hermes"]
    assert "FROM " + agent_runtime._NODE_IMAGE + " AS node-runtime" in (
        hermes_runtime.dockerfile
    )
    assert (
        "COPY --from=node-runtime /usr/local/bin/node "
        "/opt/fugue-agent-runtime/bin/node"
    ) in hermes_runtime.dockerfile
    assert "/opt/fugue-agent-runtime/lib/node_modules/npm" in (
        hermes_runtime.dockerfile
    )
    assert "npm --version | grep -F \"10.9.8\"" in hermes_runtime.dockerfile
    assert (
        "PATH=/opt/fugue-agent-runtime/bin:$PATH bash /tmp/hermes-install.sh"
        in hermes_runtime.dockerfile
    )
    probe = " ".join(hermes_runtime.probe)
    assert "node --version | grep -F v22.23.0" in probe
    assert "npm --version | grep -F 10.9.8" in probe
    assert '"disabled_toolsets": ["delegation"]' in hermes_adapter
    assert 'cat >> "$HOME/.hermes/config.yaml"' in hermes_adapter
    assert 'mkdir -p "$HOME/.hermes/skills"' in hermes_adapter
    assert "/tmp/hermes/config.yaml" not in hermes_adapter
    assert "/tmp/hermes/skills" not in hermes_adapter
    assert "ln -sf {runtime}/bin/node /usr/local/bin/node" in hermes_adapter
    assert "ln -sf {runtime}/bin/npm /usr/local/bin/npm" in hermes_adapter
    assert (
        'export PATH="/opt/fugue-agent-runtime/bin:$HOME/.local/bin:$PATH"'
        in hermes_adapter
    )
    openclaw_runtime = agent_runtime.RUNTIMES["openclaw"]
    assert openclaw_runtime.version == (
        "openclaw@2026.7.1+weave-openclaw@0.1.1+"
        "weave-otel2.1+fugue-load-path.1"
    )
    assert "npm ci --ignore-scripts" in openclaw_runtime.dockerfile
    assert "weave-openclaw/openclaw.plugin.json" in " ".join(
        openclaw_runtime.probe
    )
    openclaw_adapter = ranges["openclaw"]
    assert 'plugins.setdefault("load", {})' in openclaw_adapter
    assert "plugins list --json" in openclaw_adapter
    assert "openclaw config validate --json" in openclaw_adapter
    assert "openclaw config get mcp.servers --json" in openclaw_adapter
    assert openclaw_adapter.index("_verify_mcp_config_command()") < (
        openclaw_adapter.index("_start_gateway_command()")
    )
    assert r'plugin.status!==\"loaded\"' in openclaw_adapter
    assert r'plugin.version!==\"{self._WEAVE_PLUGIN_VERSION}\"' in openclaw_adapter
    assert "~/.openclaw/npm/projects" not in openclaw_adapter
    claude_runtime = agent_runtime.RUNTIMES["claude-code"]
    assert "npm ci --ignore-scripts" in claude_runtime.dockerfile
    assert "lib/node_modules/npm" in claude_runtime.dockerfile
    assert "bin/npm" in claude_runtime.dockerfile
    assert "lib/node_modules/weave-claude-code" in claude_runtime.dockerfile
    assert "export NPM_CONFIG_PREFIX=/opt/fugue-agent-runtime" in (
        claude_runtime.dockerfile
    )
    assert "marketplace.json" in " ".join(claude_runtime.probe)
    assert '"NPM_CONFIG_PREFIX": "/opt/fugue-agent-runtime"' in ranges["claude-code"]
    assert "export NPM_CONFIG_PREFIX=/opt/fugue-agent-runtime" in (
        ranges["claude-code"]
    )
    assert "hermes-install.sh" in agent_runtime.RUNTIMES["hermes"].dockerfile


def test_trial_mutator_lock_covers_task_image_conda_environments() -> None:
    source = Path("fugue/agents/model_plane.py").read_text()
    guard = source[
        source.index("async def _lock_trial_mutators") : source.index(
            "async def _install_context_runtime"
        )
    ]

    assert "/opt/miniconda3/envs/*/bin" in guard
    assert "/opt/conda/envs/*/bin" in guard
    for executable in ("pip", "pip3", "conda", "mamba", "micromamba"):
        assert executable in guard
    assert 'for module in pip ensurepip' in guard
    assert "Fugue trial policy blocks package installation" in guard
    assert "_apply_trial_policy_environment(public_env)" in source
    assert '"PIP_NO_INDEX": "1"' in source
    assert '"UV_OFFLINE": "1"' in source


def test_weave_codex_patch_preserves_gateway_metadata() -> None:
    source = Path("configs/fugue/runtime/codex/patch-runtime.mjs").read_text()
    assert "dist/rollout/parser.js" in source
    assert "tool.kind !== 'mcp'" in source
    assert "7c5c83f0b79d9505c3501b70fc90c96e0bf40156ca1ccd10d8442c3700e05869" in source


def test_codex_mcp_patch_flattens_only_the_model_tool_boundary() -> None:
    patch = Path("configs/fugue/runtime/codex/codex-flat-mcp.patch").read_text()
    assert "ToolSpec::Function(tool)" in patch
    assert "ToolName::plain(join_tool_name" in patch
    assert "handle_mcp_tool_call" not in patch
    assert "base_url" not in patch
    assert "mcp_servers" not in patch


def test_agent_runtime_lock_rejects_contract_drift(tmp_path: Path) -> None:
    spec = agent_runtime.RUNTIMES["codex"]
    root = tmp_path / agent_runtime.AGENT_RUNTIME_ROOT / "codex"
    root.mkdir(parents=True)
    lock = {
        "schema_version": 1,
        "harness": "codex",
        "version": spec.version,
        "recipe_sha256": spec.recipe_sha256,
        "image": spec.image,
        "image_id": "sha256:" + "a" * 64,
        "architecture": "amd64",
    }
    path = root / "runtime-lock-amd64.json"
    path.write_text(json.dumps(lock))
    assert agent_runtime.read_runtime_lock("codex", tmp_path) == lock
    lock["version"] = "drifted"
    path.write_text(json.dumps(lock))
    assert agent_runtime.read_runtime_lock("codex", tmp_path) is None


def test_agent_runtime_lock_requires_architecture_qualified_name(tmp_path: Path) -> None:
    spec = agent_runtime.RUNTIMES["codex"]
    root = tmp_path / agent_runtime.AGENT_RUNTIME_ROOT / "codex"
    root.mkdir(parents=True)
    (root / "runtime-lock.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "harness": "codex",
                "version": spec.version,
                "recipe_sha256": spec.recipe_sha256,
                "image": spec.image,
                "image_id": "sha256:" + "a" * 64,
                "architecture": "amd64",
            }
        )
    )

    assert agent_runtime.read_runtime_lock("codex", tmp_path) is None


def test_prepare_agent_runtime_records_image_identity(
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
                            "Architecture": "amd64",
                            "Os": "linux",
                        }
                    ]
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(agent_runtime.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(agent_runtime.subprocess, "run", run)
    lock = agent_runtime.prepare_runtime("codex", repo_root=tmp_path)
    assert lock["image_id"] == "sha256:" + "a" * 64
    assert commands[0][:6] == [
        "docker",
        "build",
        "--provenance=false",
        "--platform",
        "linux/amd64",
        "--pull",
    ]
    mount = agent_runtime.runtime_mount("codex", tmp_path)
    assert mount == {
        "type": "image",
        "source": lock["image_id"],
        "target": "/opt/fugue-agent-runtime",
        "read_only": True,
        "image": {"subpath": "opt/fugue-agent-runtime"},
    }


def test_prepare_agent_runtime_reuses_a_ready_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = agent_runtime.RUNTIMES["codex"]
    root = tmp_path / agent_runtime.AGENT_RUNTIME_ROOT / "codex"
    root.mkdir(parents=True)
    lock = {
        "schema_version": 1,
        "harness": "codex",
        "version": spec.version,
        "recipe_sha256": spec.recipe_sha256,
        "image": spec.image,
        "image_id": "sha256:" + "a" * 64,
        "architecture": "amd64",
    }
    (root / "runtime-lock-amd64.json").write_text(json.dumps(lock))
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                [{"Id": lock["image_id"], "Architecture": "amd64", "Os": "linux"}]
            ),
            "",
        )

    monkeypatch.setattr(agent_runtime.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(agent_runtime.subprocess, "run", run)

    assert agent_runtime.prepare_runtime("codex", repo_root=tmp_path) == lock
    assert all(command[1] != "build" for command in commands)


def test_agent_runtime_selects_native_arm64_lock(tmp_path: Path) -> None:
    spec = agent_runtime.RUNTIMES["codex"]
    root = tmp_path / agent_runtime.AGENT_RUNTIME_ROOT / "codex"
    root.mkdir(parents=True)
    lock = {
        "schema_version": 1,
        "harness": "codex",
        "version": spec.version,
        "recipe_sha256": spec.recipe_sha256,
        "image": spec.image_for("arm64"),
        "image_id": "sha256:" + "b" * 64,
        "architecture": "arm64",
    }
    (root / "runtime-lock-arm64.json").write_text(json.dumps(lock))

    assert agent_runtime.read_runtime_lock("codex", tmp_path, "arm64") == lock
    assert (
        agent_runtime.runtime_mount("codex", tmp_path, "arm64")["source"]
        == (lock["image_id"])
    )
