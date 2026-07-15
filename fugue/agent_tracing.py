from __future__ import annotations

import uuid

AGENT_NAMES = {
    "hermes": "hermes-agent",
    "openclaw": "openclaw",
    "claude-code": "claude-code",
    "codex": "codex",
    "letta": "letta",
}
CONVERSATION_NAMESPACE = uuid.UUID("218f38ca-7fe1-4db2-96e0-30f9b62c20eb")

def stable_agent_name(harness: str) -> str:
    return AGENT_NAMES.get(harness, harness)


def conversation_id(run_or_cohort_key: str) -> str:
    if not run_or_cohort_key:
        raise ValueError("conversation key cannot be empty")
    return str(uuid.uuid5(CONVERSATION_NAMESPACE, run_or_cohort_key))


def openclaw_agent_id(fugue_conversation_id: str) -> str:
    """Return the per-trial OpenClaw agent id used to isolate its trace."""
    return f"fugue-{fugue_conversation_id}"


def openclaw_conversation_id(fugue_conversation_id: str) -> str:
    """Return the conversation id emitted by OpenClaw for its main session."""
    return f"agent:{openclaw_agent_id(fugue_conversation_id)}:main"


def agent_conversation_id(harness: str, run_or_cohort_key: str) -> str:
    """Resolve the conversation identity emitted by a typed harness adapter."""
    resolved = conversation_id(run_or_cohort_key)
    if harness == "openclaw":
        return openclaw_conversation_id(resolved)
    return resolved


def agent_conversation_name(
    *, run_name: str, task_id: str, variant_id: str, trial_index: int
) -> str:
    labels = [
        value.strip()
        for value in (run_name, task_id, variant_id)
        if value and value.strip()
    ]
    labels.append(f"t{max(1, trial_index):03d}")
    return " · ".join(labels)[:256]


def normalize_trace_content(value: str | None) -> str:
    selected = str(value or "full").strip().lower()
    if selected not in {"full", "metadata"}:
        raise ValueError("trace content must be 'full' or 'metadata'")
    return selected
