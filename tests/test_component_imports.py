from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import fugue.bench.component_imports as component_imports
from fugue.bench.component_imports import (
    _mcp_process_environment,
    add_mcp_command,
    import_mcp_config,
    import_skill,
    inspect_mcp_import,
    inspect_skill_import,
    lock_mcp_import,
    lock_skill_import,
)
from fugue.bench.integrations import bind_integrations, load_integration
from fugue.bench.library import IntegrationSelection
from fugue.bench.sources import resolve_skill


def test_imports_only_one_selected_codex_mcp_server(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[mcp_servers.wandb]
command = "uvx"
args = ["--from", "wandb-mcp-server==0.3.7", "wandb-mcp-server"]
env = { WANDB_API_KEY = "${WANDB_API_KEY}", MCP_ANALYTICS_LOG_STREAM = "stderr" }

[mcp_servers.other]
command = "npx"
args = ["other@1.0.0"]
"""
    )

    draft = import_mcp_config(
        config,
        server="wandb",
        import_id="wandb-0-3-7",
        repo_root=tmp_path,
        allowed_hosts=("api.wandb.test", "trace.wandb.test"),
    )

    assert draft.command == (
        "uvx",
        "--from",
        "wandb-mcp-server==0.3.7",
        "wandb-mcp-server",
    )
    assert draft.required_env == ("WANDB_API_KEY",)
    assert draft.fixed_env == (("MCP_ANALYTICS_LOG_STREAM", "stderr"),)
    assert draft.allowed_hosts == ("api.wandb.test", "trace.wandb.test")
    assert draft.source == "repo://config.toml"
    assert "other" not in json.dumps(draft.to_dict())


def test_mcp_probe_inherits_only_operating_env_and_declared_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/tmp/unit-home")
    monkeypatch.setenv("WANDB_API_KEY", "declared-value")
    monkeypatch.setenv("UNRELATED_API_TOKEN", "must-not-cross-boundary")
    draft = component_imports.MCPImportDraftV1(
        schema_version=1,
        id="wandb",
        transport="stdio",
        command=("uvx", "wandb-mcp-server"),
        required_env=("WANDB_API_KEY",),
        fixed_env=(("MCP_ANALYTICS_LOG_STREAM", "stderr"),),
    )

    environment = _mcp_process_environment(draft)

    assert environment["PATH"] == "/usr/bin"
    assert environment["HOME"] == "/tmp/unit-home"
    assert environment["WANDB_API_KEY"] == "declared-value"
    assert environment["MCP_ANALYTICS_LOG_STREAM"] == "stderr"
    assert "UNRELATED_API_TOKEN" not in environment


def test_relocking_identical_read_only_runtime_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = component_imports.MCPImportDraftV1(
        schema_version=1,
        id="wandb",
        transport="stdio",
        command=(
            "uvx",
            "--from",
            "wandb-mcp-server==0.4.0",
            "wandb_mcp_server",
        ),
    )

    def install(
        package: str,
        executable: str,
        destination: Path,
        *,
        runtime_platform: str,
        fixed_env: tuple[tuple[str, str], ...],
    ) -> None:
        assert package == "wandb-mcp-server==0.4.0"
        assert executable == "wandb_mcp_server"
        assert runtime_platform == "linux/amd64"
        assert fixed_env == ()
        launcher = destination / "bin/server"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/bin/sh\nexit 0\n")
        launcher.chmod(0o755)
        component_imports._make_tree_read_only(destination)

    monkeypatch.setattr(
        component_imports,
        "_install_python_tool",
        install,
    )

    first = component_imports._materialize_stdio_runtime(
        draft,
        tmp_path,
        acknowledge_package_code=True,
        runtime_platform="linux/amd64",
    )
    second = component_imports._materialize_stdio_runtime(
        draft,
        tmp_path,
        acknowledge_package_code=True,
        runtime_platform="linux/amd64",
    )

    assert second == first
    runtime_root = (
        tmp_path / component_imports.MANAGED_MCP_RUNTIME_ROOT / "wandb"
    )
    assert not list(runtime_root.glob(".prepare-*"))


def test_mcp_import_rejects_literal_credentials_and_shells(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "wandb": {
                        "command": "uvx",
                        "args": [
                            "--from",
                            "wandb-mcp-server==0.3.7",
                            "wandb-mcp-server",
                        ],
                        "env": {"WANDB_API_KEY": "super-secret-value"},
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="literal value"):
        import_mcp_config(
            config,
            server="wandb",
            import_id="wandb",
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="shell-backed"):
        add_mcp_command("bad", ["sh", "-c", "server"], repo_root=tmp_path)
    with pytest.raises(ValueError, match="exact lowercase hostnames"):
        add_mcp_command(
            "bad-host",
            [
                "uvx",
                "--from",
                "wandb-mcp-server==0.3.7",
                "wandb_mcp_server",
            ],
            repo_root=tmp_path,
            allowed_hosts=("*.wandb.test",),
        )


def test_package_mcp_lock_materializes_read_only_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_mcp_command(
        "wandb-0-3-7",
        [
            "uvx",
            "--from",
            "wandb-mcp-server==0.3.7",
            "wandb-mcp-server",
        ],
        repo_root=tmp_path,
    )

    def fake_install(
        package: str,
        executable: str,
        destination: Path,
        *,
        runtime_platform: str,
        fixed_env: tuple[tuple[str, str], ...],
    ) -> None:
        assert package == "wandb-mcp-server==0.3.7"
        assert executable == "wandb-mcp-server"
        assert runtime_platform == "linux/arm64"
        assert fixed_env == ()
        target = destination / "bin" / "server"
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o555)

    monkeypatch.setattr(
        "fugue.bench.component_imports._install_python_tool", fake_install
    )
    monkeypatch.setattr(
        "fugue.bench.component_imports._managed_runtime_platform",
        lambda: "linux/arm64",
    )
    monkeypatch.setattr(
        "fugue.bench.component_imports._managed_python_probe_command",
        lambda runtime_source,
        *,
        runtime_platform,
        required_env,
        fixed_env,
        allowed_hosts: (
            "/usr/bin/docker",
            "probe",
        ),
    )
    monkeypatch.setattr(
        "fugue.bench.component_imports._probe_stdio_manifest",
        lambda draft: (
            (
                {
                    "name": "query_wandb",
                    "description": "Query W&B evidence.",
                    "input_schema": {"type": "object"},
                },
            ),
            {"name": "wandb", "version": "0.3.7"},
        ),
    )

    with pytest.raises(ValueError, match="acknowledge"):
        lock_mcp_import("wandb-0-3-7", tmp_path)
    lock = lock_mcp_import(
        "wandb-0-3-7",
        tmp_path,
        acknowledge_package_code=True,
    )

    assert lock.support == "supported"
    assert lock.runtime_platform == "linux/arm64"
    assert lock.runtime_digest and lock.runtime_digest.startswith("sha256:")
    assert lock.tool_manifest_digest
    assert lock.allowed_tools == ("query_wandb",)
    spec = load_integration("wandb-0-3-7", tmp_path)
    assert spec.runtime.type == "managed"
    binding = bind_integrations(
        [IntegrationSelection("wandb-0-3-7")],
        repo_root=tmp_path,
        runtime_root=tmp_path / ".fugue" / "jobs",
        job_name="test",
        env={},
        write=False,
    )
    assert binding.mounts[0]["read_only"] is True
    assert binding.mcp_servers[0]["command"] == (
        "/fugue-components/wandb-0-3-7/bin/server"
    )
    assert binding.provenance[0]["kind"] == "mcp"
    assert binding.provenance[0]["version_identity"].startswith("source:sha256:")
    assert binding.provenance[0]["runtime_digest"] == lock.runtime_digest
    assert binding.provenance[0]["lock_digest"] == spec.version

    runtime = Path(binding.mounts[0]["source"])
    (runtime / "bin" / "server").chmod(0o755)
    (runtime / "bin" / "server").write_text("#!/bin/sh\nexit 1\n")
    with pytest.raises(ValueError, match="digest changed"):
        bind_integrations(
            [IntegrationSelection("wandb-0-3-7")],
            repo_root=tmp_path,
            runtime_root=tmp_path / ".fugue" / "jobs",
            job_name="test",
            env={},
            write=False,
        )


def test_python_mcp_probe_is_pinned_isolated_and_secret_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(component_imports.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        component_imports.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    command = component_imports._managed_python_probe_command(
        tmp_path,
        runtime_platform="linux/arm64",
        required_env=("WANDB_API_KEY",),
        fixed_env=(("MCP_ANALYTICS_LOG_STREAM", "stderr"),),
    )

    joined = " ".join(command)
    assert "--interactive" in command
    assert "--pull never" in joined
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges" in joined
    assert "readonly" in joined
    assert "WANDB_API_KEY" in command
    assert "MCP_ANALYTICS_LOG_STREAM=stderr" in command
    assert "technical-preview" not in joined
    assert "@sha256:" in joined

    live_command = component_imports._managed_python_probe_command(
        tmp_path,
        runtime_platform="linux/arm64",
        required_env=("WANDB_API_KEY",),
        fixed_env=(),
        allowed_hosts=("api.wandb.ai",),
    )
    assert "--network bridge" in " ".join(live_command)


def test_python_mcp_runtime_installs_only_from_locked_wheelhouse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requirements = tmp_path / "requirements.lock"
    requirements.write_text("dependency==1.0.0\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "dependency-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"reviewed-wheel")
    package = tmp_path / "server-1.0.0-py3-none-any.whl"
    package.write_bytes(b"server-wheel")
    calls: list[list[str]] = []
    cleaned: list[bool] = []

    monkeypatch.setattr(
        component_imports,
        "_build_locked_wheelhouse",
        lambda requirements_lock, *, runtime_platform: (
            wheelhouse,
            lambda: cleaned.append(True),
        ),
    )
    monkeypatch.setattr(
        component_imports.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    site = tmp_path / "runtime" / "site"
    site.mkdir(parents=True)

    component_imports._install_python_distribution(
        package.as_posix(),
        site,
        runtime_platform="linux/arm64",
        requirements_lock=requirements,
    )

    assert cleaned == [True]
    assert len(calls) == 1
    command = calls[0]
    assert "--no-index" in command
    assert "--only-binary" in command
    assert "aarch64-manylinux_2_36" in command
    assert requirements.as_posix() in command
    assert package.as_posix() in command
    manifest = json.loads(
        (site.parent / "wheelhouse.lock.json").read_text(encoding="utf-8")
    )
    assert list(manifest) == [wheel.name]
    assert len(manifest[wheel.name]) == 64


def test_python_runtime_digest_ignores_installer_local_metadata(
    tmp_path: Path,
) -> None:
    digests: list[str] = []
    for index in (1, 2):
        runtime = tmp_path / f"runtime-{index}"
        site = runtime / "site"
        package = site / "example"
        metadata = site / "example-1.0.dist-info"
        scripts = site / "bin"
        package.mkdir(parents=True)
        metadata.mkdir()
        scripts.mkdir()
        (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        script = scripts / "example"
        script.write_text(
            f"#!{tmp_path}/checkout-{index}/.venv/bin/python3\n"
            "from example import VALUE\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        (metadata / "METADATA").write_text(
            "Name: example\nVersion: 1.0\n",
            encoding="utf-8",
        )
        (metadata / "uv_cache.json").write_text(
            json.dumps({"timestamp": index}),
            encoding="utf-8",
        )
        (metadata / "direct_url.json").write_text(
            json.dumps({"url": f"file:///tmp/build-{index}/example.whl"}),
            encoding="utf-8",
        )
        (metadata / "RECORD").write_text(
            "bin/example,sha256=installer-local,1\n"
            "example/__init__.py,,\n"
            "example-1.0.dist-info/METADATA,,\n"
            "example-1.0.dist-info/uv_cache.json,,\n"
            "example-1.0.dist-info/direct_url.json,,\n"
            "example-1.0.dist-info/RECORD,,\n",
            encoding="utf-8",
        )

        component_imports._normalize_python_install_metadata(site)
        digests.append(component_imports._runtime_directory_digest(runtime))

        assert not (metadata / "uv_cache.json").exists()
        assert not (metadata / "direct_url.json").exists()
        assert "uv_cache.json" not in (metadata / "RECORD").read_text()
        assert "direct_url.json" not in (metadata / "RECORD").read_text()
        assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3\n")

    assert digests[0] == digests[1]


def test_mcp_lock_identity_is_portable_across_checkout_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        component_imports,
        "_materialize_stdio_runtime",
        lambda draft, repo_root, *, acknowledge_package_code, runtime_platform: (
            "sha256:" + "a" * 64,
            (f"/fugue-components/{draft.id}/bin/server",),
        ),
    )
    monkeypatch.setattr(
        component_imports,
        "_managed_python_probe_command",
        lambda runtime_source, *, runtime_platform, required_env, fixed_env, allowed_hosts: (
            "/usr/bin/docker",
            "probe",
        ),
    )
    monkeypatch.setattr(
        component_imports,
        "_probe_stdio_manifest",
        lambda draft: (
            (
                {
                    "name": "query_wandb",
                    "description": "Query W&B evidence.",
                    "input_schema": {"type": "object"},
                },
            ),
            {"name": "wandb", "version": "0.4.0"},
        ),
    )

    locks = []
    versions = []
    for checkout_name in ("checkout-a", "checkout-b"):
        repo_root = tmp_path / checkout_name
        config = repo_root / "config" / "mcp.json"
        config.parent.mkdir(parents=True)
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "wandb": {
                            "command": "uvx",
                            "args": [
                                "--from",
                                "wandb-mcp-server==0.4.0",
                                "wandb-mcp-server",
                            ],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        draft = import_mcp_config(
            config,
            server="wandb",
            import_id="wandb-0-4",
            repo_root=repo_root,
        )
        assert draft.source == "repo://config/mcp.json"
        # Existing V1 drafts recorded the checkout's absolute config path.
        # Locking them again must migrate provenance into the portable form.
        legacy_draft = draft.to_dict()
        legacy_draft["source"] = config.resolve().as_posix()
        (repo_root / component_imports.MCP_DRAFT_ROOT / "wandb-0-4.json").write_text(
            json.dumps(legacy_draft, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lock = lock_mcp_import(
            "wandb-0-4",
            repo_root,
            acknowledge_package_code=True,
            target_platform="linux/amd64",
        )
        locks.append(lock.to_dict())
        versions.append(load_integration("wandb-0-4", repo_root).version)

    assert locks[0] == locks[1]
    assert versions[0] == versions[1]


def test_mcp_add_accepts_public_git_source_at_full_commit(tmp_path: Path) -> None:
    commit = "a2bae7271323ac43262ffb73454b0aff01ddc808"
    draft = add_mcp_command(
        "wandb-0-4",
        [
            "uvx",
            "--from",
            "git+https://github.com/wandb/wandb-mcp-server@" + commit,
            "wandb-mcp-server",
        ],
        repo_root=tmp_path,
    )

    assert draft.command[2].endswith("@" + commit)

    with pytest.raises(ValueError, match="exact name==version|full commit"):
        add_mcp_command(
            "wandb-moving",
            [
                "uvx",
                "--from",
                "git+https://github.com/wandb/wandb-mcp-server@main",
                "wandb-mcp-server",
            ],
            repo_root=tmp_path,
        )


def test_git_mcp_lock_reports_unpublished_commit_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise subprocess.CalledProcessError(128, command, stderr="unknown revision")

    monkeypatch.setattr(component_imports.subprocess, "run", run)

    with pytest.raises(
        RuntimeError,
        match="push the reviewed commit before locking it",
    ):
        component_imports._build_exact_git_wheel(
            "git+https://github.com/wandb/wandb-mcp-server@"
            "571ee85eedd73e06bd414316f73d732c398535d6"
        )


def test_remote_mcp_without_version_is_exploratory(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "url": "https://mcp.example.com/mcp",
                        "transport": "streamable-http",
                    }
                }
            }
        )
    )
    import_mcp_config(
        config,
        server="remote",
        import_id="remote",
        repo_root=tmp_path,
    )
    lock = lock_mcp_import("remote", tmp_path)
    assert lock.support == "experimental"
    assert inspect_mcp_import("remote", tmp_path)["lock"]["url"].startswith("https://")


def test_local_agent_skill_is_reviewed_locked_and_resolved(tmp_path: Path) -> None:
    source = tmp_path / "incoming" / "verify-source"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: verify-current-source\n"
        "description: Inspect the current source before answering.\n---\n\n"
        "# Verify current source\n\nOpen and cite the authoritative revision.\n"
    )

    draft = import_skill(
        source.as_posix(),
        repo_root=tmp_path,
        import_id="verify-current-source",
    )
    inspection = inspect_skill_import(draft.id, tmp_path)
    lock = lock_skill_import(draft.id, tmp_path)
    resolved = resolve_skill(draft.id, tmp_path)

    assert inspection["digest"] == lock.digest
    assert resolved.digest == lock.digest
    assert resolved.path != source
    assert resolved.path.joinpath("SKILL.md").is_file()


def test_agent_skill_rejects_symlinks_and_secret_like_files(tmp_path: Path) -> None:
    source = tmp_path / "bad-skill"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "---\nname: bad-skill\ndescription: Unsafe fixture.\n---\n"
    )
    (source / "api-token.txt").write_text("secret")
    import_skill(source.as_posix(), repo_root=tmp_path, import_id="bad-skill")
    with pytest.raises(ValueError, match="secret"):
        inspect_skill_import("bad-skill", tmp_path)
