from __future__ import annotations

from copy import deepcopy

import pytest

from fugue.bench.candidates import resolve_candidate
from fugue.bench.job_config import _comparison_example_id


def _candidate_inputs() -> dict:
    return {
        "harness": "codex",
        "model_route": {"provider": "openai", "model_id": "gpt-5"},
        "prompt_digest": "prompt-a",
        "skills": [{"id": "reviewed", "sha256": "skill-a"}],
        "context": {
            "id": "none",
            "version": "1",
            "config_hash": "context-a",
            "delivery": "portable",
        },
        "integrations": [{"id": "search", "version": "1"}],
        "agent": {"agent_kwargs": {"reasoning": "high"}},
        "execution": {
            "harbor_version": "0.18.0",
            "trace_content": "full",
        },
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("harness", "claude-code"),
        ("model_route", {"provider": "anthropic", "model_id": "claude"}),
        ("prompt_digest", "prompt-b"),
        ("skills", [{"id": "reviewed", "sha256": "skill-b"}]),
        (
            "context",
            {
                "id": "none",
                "version": "1",
                "config_hash": "context-a",
                "delivery": "native_mcp",
            },
        ),
        ("integrations", [{"id": "browser", "version": "1"}]),
        ("agent", {"agent_kwargs": {"reasoning": "low"}}),
    ),
)
def test_every_behavioral_input_changes_candidate_identity(
    field: str, replacement: object
) -> None:
    original = _candidate_inputs()
    changed = deepcopy(original)
    changed[field] = replacement

    assert resolve_candidate(**original).candidate_id != resolve_candidate(
        **changed
    ).candidate_id


def test_execution_policy_changes_fingerprint_not_candidate_identity() -> None:
    original = _candidate_inputs()
    changed = deepcopy(original)
    changed["execution"]["trace_content"] = "metadata"

    first = resolve_candidate(**original)
    second = resolve_candidate(**changed)

    assert first.candidate_id == second.candidate_id
    assert first.execution_fingerprint != second.execution_fingerprint


def test_resolved_candidate_does_not_expose_mutable_internal_state() -> None:
    resolved = resolve_candidate(**_candidate_inputs())
    definition = resolved.definition
    definition["harness"] = "mutated"

    assert resolved.definition["harness"] == "codex"


def test_comparison_example_identity_excludes_trial_index() -> None:
    identity = _comparison_example_id(
        dataset_id="dataset", workload_id="workload", task_id="task"
    )

    assert identity == _comparison_example_id(
        dataset_id="dataset", workload_id="workload", task_id="task"
    )
    assert len(identity) == 64
