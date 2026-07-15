from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("harbor.agents.installed.codex")

from fugue.agents.model_plane import FugueCodex


def test_codex_runtime_uses_cell_home_and_structured_mcp_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(FugueCodex)
    agent.mcp_servers = [
        SimpleNamespace(
            name="repo",
            transport="stdio",
            command="python",
            args=["-m", "server"],
        )
    ]
    monkeypatch.setenv("FUGUE_RUN_ID", "run-a")
    monkeypatch.setenv("FUGUE_TASK_NAME", "task-a")
    monkeypatch.setenv("FUGUE_HARNESS", "codex")
    monkeypatch.setenv("FUGUE_CONTEXT_SYSTEM_ID", "gitnexus")
    monkeypatch.setenv("FUGUE_CONTEXT_CONFIG_HASH", "config-a")
    monkeypatch.setenv("FUGUE_VARIANT_ID", "gitnexus")
    monkeypatch.setenv("FUGUE_TRIAL_INDEX", "1")

    home = agent._codex_home()
    command = agent._build_register_mcp_servers_command()

    assert home.as_posix().startswith("/tmp/fugue-codex/")
    assert home.as_posix() != "/tmp/codex-home"
    assert command is not None
    assert 'command = "python"' in command
    assert 'args = ["-m","server"]' in command

    registration = agent._context_registration(
        {"status": "registered", "transport": "native_mcp"}
    )
    assert registration["context_system_id"] == "gitnexus"
    assert registration["registration_digest"].startswith("sha256:")
