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
from fugue.bench.candidates import stable_digest
from fugue.bench.execution import CellOutcome, PlannedCell
from fugue.bench.export import GeneratedEvaluationCoordinator

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


def _materialize_native_local_transcript(
    logs_dir: Path,
    *,
    harness: str,
    session_id: str,
) -> None:
    if harness == "hermes":
        (logs_dir / "hermes-session.jsonl").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "role": "assistant",
                    "content": "completed",
                    "tool_calls": [{"name": "read_file", "arguments": {}}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return
    if harness == "openclaw":
        (logs_dir / "openclaw.txt").write_text(
            json.dumps(
                {
                    "sessionId": session_id,
                    "meta": {
                        "agentMeta": {
                            "sessionFile": "/home/agent/.openclaw/session.jsonl"
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (logs_dir / "openclaw.session.jsonl").write_text(
            json.dumps(
                {
                    "type": "message",
                    "message": {"role": "assistant", "content": "completed"},
                }
            )
            + "\n"
            + json.dumps(
                {"type": "tool_call", "name": "read_file", "arguments": {}}
            )
            + "\n",
            encoding="utf-8",
        )
        return
    if harness == "claude-code":
        (logs_dir / "claude-code.txt").write_text(
            json.dumps(
                {"type": "system", "subtype": "init", "session_id": session_id}
            )
            + "\n"
            + json.dumps(
                {
                    "type": "assistant",
                    "session_id": session_id,
                    "message": {
                        "content": [{"type": "text", "text": "completed"}]
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        transcript = (
            logs_dir / "sessions" / "projects" / "-app" / f"{session_id}.jsonl"
        )
        transcript.parent.mkdir(parents=True)
        transcript.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "sessionId": session_id,
                    "message": {
                        "content": [{"type": "tool_use", "name": "Read", "input": {}}]
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return
    if harness == "codex":
        (logs_dir / "codex.txt").write_text(
            json.dumps({"type": "thread.started", "thread_id": session_id}) + "\n",
            encoding="utf-8",
        )
        transcript = (
            logs_dir
            / "sessions"
            / "2026"
            / "08"
            / "18"
            / f"rollout-2026-08-18T12-00-00-{session_id}.jsonl"
        )
        transcript.parent.mkdir(parents=True)
        transcript.write_text(
            json.dumps({"type": "session_meta", "payload": {"id": session_id}})
            + "\n"
            + json.dumps(
                {"type": "function_call", "name": "read_file", "arguments": {}}
            )
            + "\n",
            encoding="utf-8",
        )
        return
    raise AssertionError(f"unsupported harness: {harness}")


@pytest.mark.parametrize(
    ("harness", "harness_type"),
    [
        ("hermes", FugueHermes),
        ("openclaw", FugueOpenClaw),
        ("claude-code", FugueClaudeCode),
        ("codex", FugueCodex),
    ],
)
def test_local_harness_run_closes_provider_neutral_evidence_manifest(
    harness: str,
    harness_type: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / harness
    trial_dir = root / "jobs" / "cell-a"
    logs_dir = trial_dir / "agent"
    run_id = f"run-{harness}"
    session_id = "11111111-2222-4333-8444-555555555555"
    cell = PlannedCell(
        id="cell-a",
        run_id=run_id,
        run_name=run_id,
        workload_id="suite",
        task_id="task-a",
        harness=harness,
        context_system_id="none",
        variant_id="baseline",
        model_provider="anthropic",
        model="anthropic/claude-test",
        trial_index=1,
        comparison_example_id="example-a",
        candidate_id="candidate-a",
        execution_fingerprint="execution-a",
        config_path=root / "config.json",
        result_path=trial_dir / "result.json",
        command=("harbor", "run"),
        env={"FUGUE_EVIDENCE_MODE": "local", "FUGUE_DATASET": "suite@v1"},
        n_attempts=1,
        evaluation_asset_lock_sha256="e" * 64,
        run_snapshot_sha256="a" * 64,
    )
    conformance = {
        "status": "passed",
        "execution_identity": {"status": "passed", "digest": "x" * 64},
        "local_artifact_privacy_scan": {
            "status": "passed",
            "matches": 0,
            "redaction_pattern_detector": "fugue.redaction.redact_text",
        },
        "private_label_boundary": {"status": "passed"},
        "docker_cleanup": {"status": "passed", "matched_containers": []},
    }
    coordinator = GeneratedEvaluationCoordinator(
        [cell],
        repo_root=root,
        env={"ANTHROPIC_API_KEY": "not-a-real-secret-value"},
        cell_conformance=lambda _cell: conformance,
        require_complete_evidence=True,
        evidence_mode="local",
    )
    overlay = coordinator.begin_cell(cell)
    assert overlay is not None

    environment = {
        "FUGUE_RUN_ID": run_id,
        "FUGUE_WORKLOAD_ID": "suite",
        "FUGUE_TASK_NAME": "task-a",
        "FUGUE_HARNESS": harness,
        "FUGUE_CONTEXT_SYSTEM_ID": "none",
        "FUGUE_VARIANT_ID": "baseline",
        "FUGUE_TRIAL_INDEX": "1",
        "FUGUE_COMPARISON_EXAMPLE_ID": "example-a",
        "FUGUE_CANDIDATE_ID": "candidate-a",
        "FUGUE_EXECUTION_FINGERPRINT": "execution-a",
        "FUGUE_DATASET": "suite@v1",
        "ANTHROPIC_API_KEY": "model-key",
        **overlay,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    for name in _TRACE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    agent = _local_agent(harness_type, logs_dir, monkeypatch)
    agent._context_registration_meta = {"status": "not_assigned"}
    calls: list[dict[str, Any]] = []

    async def exec_as_agent(
        _environment: object,
        command: str,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        calls.append({"command": command, "env": dict(env or {}), **kwargs})
        if any(
            marker in command
            for marker in (
                "hermes --yolo chat",
                "openclaw agent --local --json",
                "claude --verbose --output-format=stream-json",
                "codex exec --dangerously-bypass-approvals-and-sandbox",
                "_openclaw_container_copy_session_transcript",
            )
        ):
            _materialize_native_local_transcript(
                logs_dir,
                harness=harness,
                session_id=session_id,
            )
        stdout = ""
        if "tail -c" in command:
            stdout = (logs_dir / "claude-code.txt").read_text(encoding="utf-8")
        elif "openclaw agent --local --json" in command:
            stdout = json.dumps({"sessionId": session_id}) + "\n"
        else:
            stdout = "native agent output"
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    async def begin_trial(
        observed_harness: str,
        route: object,
        _environment: object,
    ) -> None:
        agent._meta_begin(observed_harness, route)

    async def finish_trial(_environment: object) -> None:
        agent._meta_end()

    agent.exec_as_agent = exec_as_agent
    agent._begin_trial = begin_trial
    agent._finish_trial = finish_trial
    agent._verify_skill_registration = AsyncMock()
    agent._lock_trial_mutators = AsyncMock()
    agent._install_tool_result_guard = AsyncMock()
    agent._install_action_gate = AsyncMock()
    agent._detect_home = AsyncMock(return_value="/home/agent")
    agent._task_interaction = lambda _instruction: _Interaction()
    agent._set_task_interaction_summary = lambda _interaction: None

    asyncio.run(agent.run("perform the task", object(), SimpleNamespace()))

    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "suite/task-a",
                "trial_name": "cell-a",
                "agent_result": {
                    "n_input_tokens": 12,
                    "n_output_tokens": 3,
                    "cost_usd": 0.02,
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
                "started_at": "2026-08-18T12:00:00Z",
                "finished_at": "2026-08-18T12:00:02Z",
            }
        ),
        encoding="utf-8",
    )
    row = coordinator.finish_cell(
        cell,
        CellOutcome(cell.id, "passed", returncode=0),
    )
    conformance_receipt = {
        "schema_version": 2,
        "run_id": run_id,
        "backend": "local_harbor_docker",
        "status": "passed",
        "enforced": True,
        "receipt_sha256": "",
    }
    conformance_receipt["receipt_sha256"] = stable_digest(conformance_receipt)
    (root / f".fugue/runtime/{run_id}/harbor-conformance.json").write_text(
        json.dumps(conformance_receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = coordinator.finalize()

    assert row is not None
    assert manifest is not None and manifest.status == "complete"
    assert len(manifest.attempt_records) == 1
    assert {link["system"] for link in row["local_evidence_links"]} == {
        "local_artifact"
    }
    assert {link["kind"] for link in row["local_evidence_links"]} == {
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_root",
        "dataset",
    }
    agent_receipt_path = next(
        (root / f".fugue/runtime/{run_id}/evidence/agents").glob("*.json")
    )
    receipt = json.loads(agent_receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "resolved"
    assert receipt["native_weave_call"] is False
    assert receipt["correlation_method"] == "isolated_trial_directory_v1"
    assert receipt["primary_session_id"] == session_id
    assert receipt["tool_event_count"] == 1
    assert len(receipt["tool_events_sha256"]) == 64
    assert all(_TRACE_ENV_NAMES.isdisjoint(call["env"]) for call in calls)
