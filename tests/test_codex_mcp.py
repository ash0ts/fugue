from __future__ import annotations

from types import SimpleNamespace

import pytest

from fugue.codex_mcp import render_codex_mcp_toml


def test_codex_mcp_toml_keeps_command_and_arguments_separate() -> None:
    rendered = render_codex_mcp_toml(
        [
            SimpleNamespace(
                name="fugue-context",
                transport="stdio",
                command="python",
                args=["-m", "fugue.mcp_proxy", "--", "npx", "server@1.2.3"],
            )
        ]
    )

    assert 'command = "python"' in rendered
    assert 'args = ["-m","fugue.mcp_proxy","--","npx","server@1.2.3"]' in rendered
    assert 'command = "python -m' not in rendered


def test_codex_mcp_toml_rejects_unsafe_names_and_transports() -> None:
    with pytest.raises(ValueError, match="invalid Codex MCP server name"):
        render_codex_mcp_toml(
            [SimpleNamespace(name="bad.name", transport="stdio", command="x")]
        )
    with pytest.raises(ValueError, match="unsupported transport"):
        render_codex_mcp_toml(
            [SimpleNamespace(name="safe", transport="portable", command="x")]
        )
