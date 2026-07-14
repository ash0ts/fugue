from __future__ import annotations

import uuid

import pytest

from fugue.agent_tracing import (
    conversation_id,
    normalize_trace_content,
    stable_agent_name,
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


def test_trace_content_is_explicit() -> None:
    assert normalize_trace_content(None) == "full"
    assert normalize_trace_content(" metadata ") == "metadata"
    with pytest.raises(ValueError, match="full.*metadata"):
        normalize_trace_content("redacted")
