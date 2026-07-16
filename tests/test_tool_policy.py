from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath

from fugue.model_plane import resolve_model_route
from fugue.tool_policy import (
    TOOL_RESULT_GUARD_PATH,
    tool_result_guard_cli_flags,
    tool_result_guard_config,
    tool_result_guard_install_command,
    tool_result_guard_script,
)


def _guard(event: dict[str, object]) -> dict[str, object] | None:
    result = subprocess.run(
        ["python3", "-c", tool_result_guard_script()],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def test_text_only_route_blocks_media_without_blocking_text_reads() -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})

    assert route.tool_result_modalities == ("text",)
    denial = _guard(
        {"tool_name": "Read", "tool_input": {"file_path": "/workspace/file.pdf"}}
    )
    assert denial == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "This model route accepts text tool results only. Use bounded text "
                "extraction or file metadata instead of sending image or PDF bytes."
            ),
        }
    }
    assert _guard(
        {"tool_name": "Read", "tool_input": {"file_path": "/workspace/file.py"}}
    ) is None
    assert _guard(
        {"tool_name": "view_image", "tool_input": {"path": "/workspace/file.png"}}
    ) == denial


def test_visual_routes_do_not_install_a_guard() -> None:
    route = resolve_model_route("openai/gpt-5", {})

    assert route.tool_result_modalities == ("text", "image")
    assert tool_result_guard_config(route, "codex") is None
    assert (
        tool_result_guard_install_command(
            route, "codex", PurePosixPath("/tmp/codex-home/hooks.json")
        )
        is None
    )
    assert tool_result_guard_cli_flags(route, "codex") == ()


def test_each_harness_uses_the_shared_guard_with_its_native_matcher() -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    claude = tool_result_guard_config(route, "claude-code")
    codex = tool_result_guard_config(route, "codex")

    assert claude is not None and codex is not None
    assert claude["hooks"]["PreToolUse"][0]["matcher"] == "^Read$"
    assert codex["hooks"]["PreToolUse"][0]["matcher"] == "^view_image$"
    for config in (claude, codex):
        command = config["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        assert command == f"python3 {TOOL_RESULT_GUARD_PATH}"
    assert tool_result_guard_cli_flags(route, "codex") == (
        "--dangerously-bypass-hook-trust",
    )
    assert tool_result_guard_cli_flags(route, "claude-code") == ()


def test_guard_installer_preserves_existing_harness_settings(tmp_path: Path) -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({"permissions": {"defaultMode": "bypass"}}))
    command = tool_result_guard_install_command(
        route,
        "claude-code",
        PurePosixPath(config_path.as_posix()),
    )

    assert command is not None
    subprocess.run(["sh", "-c", command], check=True)

    config = json.loads(config_path.read_text())
    assert config["permissions"] == {"defaultMode": "bypass"}
    assert config["hooks"]["PreToolUse"][0]["matcher"] == "^Read$"
    assert Path(TOOL_RESULT_GUARD_PATH).read_text() == tool_result_guard_script()
