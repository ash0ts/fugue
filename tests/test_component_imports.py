from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.bench.component_imports import (
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
env = { WANDB_API_KEY = "${WANDB_API_KEY}" }

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
    )

    assert draft.command == (
        "uvx",
        "--from",
        "wandb-mcp-server==0.3.7",
        "wandb-mcp-server",
    )
    assert draft.required_env == ("WANDB_API_KEY",)
    assert "other" not in json.dumps(draft.to_dict())


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

    def fake_install(package: str, executable: str, destination: Path) -> None:
        assert package == "wandb-mcp-server==0.3.7"
        assert executable == "wandb-mcp-server"
        target = destination / "bin" / "server"
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o555)

    monkeypatch.setattr(
        "fugue.bench.component_imports._install_python_tool", fake_install
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
