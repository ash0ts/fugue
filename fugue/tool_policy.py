from __future__ import annotations

import shlex
import textwrap
from pathlib import PurePosixPath
from typing import Any, Literal

from fugue.model_plane import ModelRoute

HarnessToolPolicy = Literal["claude-code", "codex"]

TOOL_RESULT_GUARD_PATH = PurePosixPath("/tmp/fugue-tool-result-guard.py")

_GUARD_SCRIPT = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json
    import pathlib
    import sys

    MEDIA_SUFFIXES = {
        ".avif", ".bmp", ".gif", ".heic", ".heif", ".ico", ".jpeg",
        ".jpg", ".pdf", ".png", ".svg", ".tif", ".tiff", ".webp",
    }
    event = json.load(sys.stdin)
    tool_name = str(event.get("tool_name") or "")
    tool_input = event.get("tool_input")
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    path = str(tool_input.get("file_path") or tool_input.get("path") or "")

    deny = tool_name == "view_image" or (
        tool_name == "Read" and pathlib.PurePath(path).suffix.lower() in MEDIA_SUFFIXES
    )
    if deny:
        reason = (
            "This model route accepts text tool results only. Use bounded text "
            "extraction or file metadata instead of sending image or PDF bytes."
        )
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }, separators=(",", ":")))
    """
)

_MATCHERS: dict[HarnessToolPolicy, str] = {
    "claude-code": "^Read$",
    "codex": "^view_image$",
}


def tool_result_guard_config(
    route: ModelRoute, harness: HarnessToolPolicy
) -> dict[str, Any] | None:
    if "image" in route.tool_result_modalities:
        return None
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": _MATCHERS[harness],
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {TOOL_RESULT_GUARD_PATH}",
                            "timeout": 5,
                        }
                    ],
                }
            ]
        }
    }


def tool_result_guard_cli_flags(
    route: ModelRoute, harness: HarnessToolPolicy
) -> tuple[str, ...]:
    if harness != "codex" or tool_result_guard_config(route, harness) is None:
        return ()
    # Codex 0.143 skips untrusted hooks in headless runs. This trusts only the
    # guard Fugue just wrote into the trial's isolated CODEX_HOME.
    return ("--dangerously-bypass-hook-trust",)


def tool_result_guard_script() -> str:
    return _GUARD_SCRIPT


def tool_result_guard_install_command(
    route: ModelRoute,
    harness: HarnessToolPolicy,
    config_path: PurePosixPath,
) -> str | None:
    hook_config = tool_result_guard_config(route, harness)
    if hook_config is None:
        return None
    installer = textwrap.dedent(
        f"""\
        import json
        from pathlib import Path

        guard = Path({str(TOOL_RESULT_GUARD_PATH)!r})
        guard.write_text({tool_result_guard_script()!r})
        guard.chmod(0o700)

        config_path = Path({str(config_path)!r})
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if not isinstance(config, dict):
                raise ValueError("tool hook config must be a JSON object")
        else:
            config = {{}}
        incoming = {hook_config!r}
        hooks = config.setdefault("hooks", {{}})
        if not isinstance(hooks, dict):
            raise ValueError("hooks must be a JSON object")
        events = hooks.setdefault("PreToolUse", [])
        if not isinstance(events, list):
            raise ValueError("hooks.PreToolUse must be a JSON array")
        for group in incoming["hooks"]["PreToolUse"]:
            if group not in events:
                events.append(group)
        temporary = config_path.with_name(config_path.name + ".tmp")
        temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\\n")
        temporary.replace(config_path)
        """
    )
    return f"python3 -c {shlex.quote(installer)}"
