from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor.agents.installed.codex")

from fugue.agents.model_plane import (
    FugueClaudeCode,
    FugueCodex,
    FugueHermes,
    FugueOpenClaw,
)

_HARNESS_TYPES = (FugueHermes, FugueOpenClaw, FugueClaudeCode, FugueCodex)
_TRACE_ENV_NAMES = {
    "FUGUE_WEAVE_API_KEY",
    "FUGUE_WEAVE_CONVERSATION_ID",
    "FUGUE_WEAVE_SINGLE_TURN_KEY",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_RESOURCE_ATTRIBUTES",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WEAVE_PROJECT",
}


class _Interaction:
    plan = SimpleNamespace(follow_up_count=0)

    def observe_agent(self, _value: str) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {"status": "complete", "follow_up_count": 0}


def _local_agent(
    harness_type: type,
    logs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FUGUE_EVIDENCE_MODE", "local")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "model-key")
    for name in _TRACE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return harness_type(
        logs_dir=logs_dir,
        model_name="anthropic/claude-test",
    )


@pytest.mark.parametrize("harness_type", _HARNESS_TYPES)
def test_local_harness_construction_requires_only_the_model_credential(
    harness_type: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _local_agent(harness_type, tmp_path / harness_type.__name__, monkeypatch)

    assert agent.model_route.provider == "anthropic"


def test_unknown_evidence_mode_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FUGUE_EVIDENCE_MODE", "sometimes")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "model-key")

    with pytest.raises(ValueError, match="local.*weave_required"):
        FugueClaudeCode(
            logs_dir=tmp_path / "claude-code",
            model_name="anthropic/claude-test",
        )


def test_legacy_default_still_requires_weave_evidence_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FUGUE_EVIDENCE_MODE", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "model-key")
    for name in _TRACE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="FUGUE_WEAVE_API_KEY"):
        FugueClaudeCode(
            logs_dir=tmp_path / "claude-code",
            model_name="anthropic/claude-test",
        )


def _capture_local_run(
    agent: Any,
    *,
    forbidden_async_methods: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def exec_as_agent(
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        calls.append({"command": command, "env": dict(env or {}), **kwargs})
        if "tail -c" in command:
            stdout = '{"session_id":"native-claude-session"}\n'
        elif "openclaw agent" in command:
            stdout = '{"sessionId":"native-openclaw-session"}\n'
        else:
            stdout = "native agent output"
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    agent.exec_as_agent = exec_as_agent
    agent._begin_trial = AsyncMock()
    agent._finish_trial = AsyncMock()
    agent._verify_skill_registration = AsyncMock()
    agent._lock_trial_mutators = AsyncMock()
    agent._install_tool_result_guard = AsyncMock()
    agent._install_action_gate = AsyncMock()
    agent._detect_home = AsyncMock(return_value="/home/agent")
    agent._copy_openclaw_session_file_to_agent_logs = AsyncMock()
    agent._task_interaction = lambda _instruction: _Interaction()
    agent._set_task_interaction_summary = lambda _interaction: None
    for method_name in forbidden_async_methods:
        setattr(
            agent,
            method_name,
            AsyncMock(
                side_effect=AssertionError(f"called {method_name} in local mode")
            ),
        )

    asyncio.run(agent.run("perform the task", object(), SimpleNamespace()))
    return calls


@pytest.mark.parametrize(
    ("harness_type", "forbidden_runtime_component"),
    [
        (FugueHermes, "hermes-otel"),
        (FugueOpenClaw, "weave-openclaw"),
        (FugueClaudeCode, "weave-claude-code"),
        (FugueCodex, "weave-codex"),
    ],
)
def test_local_install_verifies_only_the_native_harness_runtime(
    harness_type: type,
    forbidden_runtime_component: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _local_agent(harness_type, tmp_path / harness_type.__name__, monkeypatch)
    commands: list[str] = []

    async def capture(
        _environment: object,
        command: str,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    agent.exec_as_root = capture
    agent.exec_as_agent = capture
    agent._capture_runtime_fingerprint = AsyncMock()

    asyncio.run(agent.install(object()))

    rendered = "\n".join(commands)
    assert forbidden_runtime_component not in rendered
    assert f"bin/{agent.TRACE_HARNESS.split('-')[0]}" in rendered or (
        harness_type is FugueOpenClaw and "bin/openclaw" in rendered
    )


@pytest.mark.parametrize(
    ("harness_type", "native_command", "forbidden_command", "forbidden_methods"),
    [
        (
            FugueHermes,
            "hermes --yolo chat",
            "hermes_otel",
            ("_configure_hermes_otel",),
        ),
        (
            FugueOpenClaw,
            "openclaw agent --local --json",
            "weave-openclaw",
            (),
        ),
        (
            FugueClaudeCode,
            "claude --verbose --output-format=stream-json",
            "weave-claude-code",
            ("_install_weave_plugin", "_finalize_weave_session"),
        ),
        (
            FugueCodex,
            "codex exec --dangerously-bypass-approvals-and-sandbox",
            "weave-codex",
            (),
        ),
    ],
)
def test_local_harnesses_use_native_commands_without_hosted_trace_configuration(
    harness_type: type,
    native_command: str,
    forbidden_command: str,
    forbidden_methods: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _local_agent(harness_type, tmp_path / harness_type.__name__, monkeypatch)
    agent.mcp_servers = [
        SimpleNamespace(
            name="reference",
            transport="stdio",
            command="reference-mcp",
            args=["serve"],
            url=None,
        )
    ]
    calls = _capture_local_run(agent, forbidden_async_methods=forbidden_methods)

    commands = "\n".join(str(call["command"]) for call in calls)
    assert native_command in commands
    assert forbidden_command not in commands
    assert "openclaw gateway" not in commands
    assert "plugins list --json" not in commands
    for call in calls:
        assert _TRACE_ENV_NAMES.isdisjoint(call["env"])


def test_local_meta_is_provider_neutral_and_retains_native_session_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _local_agent(FugueOpenClaw, tmp_path / "agent", monkeypatch)
    monkeypatch.setenv("FUGUE_RUN_ID", "run-local")
    monkeypatch.setenv("FUGUE_TASK_NAME", "task-local")
    monkeypatch.setenv("FUGUE_TRIAL_INDEX", "1")
    monkeypatch.setenv("FUGUE_CONTEXT_SYSTEM_ID", "none")
    agent._context_registration_meta = {"status": "not_assigned"}
    agent._meta_begin("openclaw", agent.model_route)
    (agent.logs_dir / "openclaw.txt").write_text(
        '{"sessionId":"native-openclaw-session"}\n', encoding="utf-8"
    )
    agent._meta_end()

    meta = json.loads((agent.logs_dir / "fugue-meta.json").read_text())
    assert meta["evidence_mode"] == "local"
    assert meta["agent_name"] == "openclaw"
    assert meta["native_session_ids"] == ["native-openclaw-session"]
    assert meta["weave_entity"] is None
    assert meta["weave_project"] is None
    assert meta["weave_conversation_ids"] == []
    assert meta["conversation_correlation"]["status"] == ("isolated_trial_directory")
