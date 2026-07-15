from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from fugue.agent_tracing import (
    agent_conversation_id,
    agent_conversation_name,
    conversation_id,
    normalize_trace_content,
    openclaw_agent_id,
    openclaw_conversation_id,
    stable_agent_name,
)

AGENT_MODEL_PLANE = (
    Path(__file__).resolve().parents[1] / "fugue" / "agents" / "model_plane.py"
)


def test_harness_agents_are_stable_and_trials_are_deterministic() -> None:
    assert stable_agent_name("hermes") == "hermes-agent"
    assert stable_agent_name("openclaw") == "openclaw"
    assert stable_agent_name("claude-code") == "claude-code"
    assert stable_agent_name("codex") == "codex"
    first = conversation_id("run:task:codex:trial-1")
    assert first == conversation_id("run:task:codex:trial-1")
    assert first != conversation_id("run:task:codex:trial-2")
    assert str(uuid.UUID(first)) == first


def test_openclaw_trial_identity_preserves_stable_agent_name() -> None:
    run_key = "run:task:openclaw:trial-1"
    fugue_id = "872b3077-0f62-544d-8e09-aa437b84f029"

    assert openclaw_agent_id(fugue_id) == f"fugue-{fugue_id}"
    assert openclaw_conversation_id(fugue_id) == f"agent:fugue-{fugue_id}:main"
    assert agent_conversation_id("openclaw", run_key) == openclaw_conversation_id(
        conversation_id(run_key)
    )
    assert agent_conversation_id("hermes", run_key) == conversation_id(run_key)
    assert stable_agent_name("openclaw") == "openclaw"


def test_conversation_names_are_human_readable_and_bounded() -> None:
    assert agent_conversation_name(
        run_name="standup-skills-full",
        task_id="paper-anonymizer",
        variant_id="with-pdf-skill",
        trial_index=1,
    ) == "standup-skills-full · paper-anonymizer · with-pdf-skill · t001"
    assert len(
        agent_conversation_name(
            run_name="x" * 300,
            task_id="task",
            variant_id="baseline",
            trial_index=0,
        )
    ) == 256


def test_model_plane_uses_the_typed_trace_conversation_hook() -> None:
    source = AGENT_MODEL_PLANE.read_text()

    for harness in ("hermes", "openclaw", "claude-code", "codex", "letta"):
        assert f'TRACE_HARNESS = "{harness}"' in source
    assert '"gen_ai.conversation.id": self.trace_conversation_id' in source
    assert '"weave.conversation.name": agent_conversation_name(' in source
    assert '"fugue.conversation_id": self.trace_conversation_id' in source
    assert '"planned_conversation_id": self.trace_conversation_id' in source
    assert '"FUGUE_WEAVE_CONVERSATION_ID": self.conversation_id' not in source
    assert "dict.fromkeys([self.trace_conversation_id, *native_ids])" in source


def test_model_plane_normalizes_only_declared_artifact_transport() -> None:
    source = AGENT_MODEL_PLANE.read_text()

    assert 'FUGUE_EXPECTED_ARTIFACT_PATHS' in source
    assert "artifact_recoveries(expected, repo_root)" in source
    assert '"artifact_normalization"' in source


def test_trace_content_is_explicit() -> None:
    assert normalize_trace_content(None) == "full"
    assert normalize_trace_content(" metadata ") == "metadata"
    with pytest.raises(ValueError, match="full.*metadata"):
        normalize_trace_content("redacted")


def test_trial_trace_attributes_are_flat_and_comparable() -> None:
    source = AGENT_MODEL_PLANE.read_text()

    for attribute in (
        "fugue.run_id",
        "fugue.experiment_id",
        "fugue.workload_id",
        "fugue.harness",
        "fugue.variant_id",
        "fugue.context_system_id",
        "fugue.context_delivery",
        "fugue.context_registration_status",
        "fugue.context_support",
        "fugue.integration_ids",
        "fugue.task_id",
        "fugue.trial_index",
        "fugue.comparison_example_id",
        "fugue.candidate_id",
        "fugue.evaluation_scope_id",
        "fugue.model_provider",
        "fugue.model",
        "fugue.tool_result_modalities",
        "weave.eval.predict_and_score_call_id",
        "weave.eval.project_id",
        "weave.eval.evaluation_name",
    ):
        assert f'"{attribute}"' in source
    assert "FUGUE_TRACE_ATTRIBUTES_JSON" in source
    assert "key: str(value)" in source


def test_hermes_staging_promotes_resource_attributes_to_spans() -> None:
    source = AGENT_MODEL_PLANE.read_text()

    assert "self.config.resource_attributes or {}" in source
    assert "hermes-otel span attribute patch target was not found" in source


def test_native_plugin_patches_are_pinned_and_integrity_checked() -> None:
    source = AGENT_MODEL_PLANE.read_text()

    assert source.count('_WEAVE_PLUGIN_VERSION = "0.1.1"') == 2
    assert '_WEAVE_PLUGIN_VERSION = "0.2.12"' in source
    assert '_HERMES_VERSION = "v2026.6.5"' in source
    assert '_OPENCLAW_VERSION = "2026.7.1"' in source
    assert '_CLAUDE_CODE_VERSION = "2.1.210"' in source
    assert '_CODEX_VERSION = "0.143.0"' in source
    assert "@latest" not in source
    assert "openclaw plugins install weave-openclaw@" in source
    assert "{self._HERMES_VERSION}/scripts/install.sh" in source
    assert "--skip-browser --no-skills --non-interactive" in source
    assert "npm install -g weave-claude-code@" in source
    assert "weave-codex@" in source
    assert "weave-codex run -- codex exec" in source
    assert "weave-codex install" not in source
    assert "tool_result_guard_cli_flags(self.model_route, \"codex\")" in source
    assert "set -o pipefail" in source
    assert "emitter pattern missing" in source
    assert "baggage pattern missing" in source
    assert "processor pattern missing" in source
    assert "key.startsWith('fugue.')" in source
    assert "self._resolved_env_vars.update(" in source
    assert "expected 3 span objects" in source
    assert "codex mcp list --json" in source
    assert "pending_native_registration" in source
    assert '"status": "failed"' in source
    for tool in (
        "web_search",
        "web_fetch",
        "browser",
        "image",
        "memory_search",
        "memory_get",
    ):
        assert f'"{tool}"' in source
    assert "context registration probe failed before agent execution" in source
