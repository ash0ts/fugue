from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DIRECT_SCORE_FIELDS = (
    "reward",
    "mrr",
    "ndcg_at_10",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "evidence_recall",
    "citation_correctness",
    "fact_recall",
    "judge_correctness",
    "judge_completeness",
    "judge_groundedness",
    "judge_overall",
)

GENERATED_EVALUATION_SCORE_DIMENSIONS = (
    "task_completion",
    "correctness",
    "groundedness",
    "tool_use",
    "artifact_quality",
)

PROMPT_INJECTION_REWARDS = (
    "safe_and_useful",
    "safe_but_failed_or_refused",
    "compromised",
    "incorrect",
    "task_complete",
    "false_positive_refusal",
    "evidence_preserved",
)

PROMPT_INJECTION_OPTIONAL_REWARDS = (
    "attack_encountered",
    "sensitive_action_attempted",
    "action_gate_blocked",
    "action_gate_allowed",
)

EVIDENCE_USE_REWARDS = (
    "artifact_schema_valid",
    "answer_facts_correct",
    "current_document_cited",
    "current_document_used",
    "unsupported_claims_absent",
)

BEHAVIORAL_TERMINAL_KINDS = frozenset(
    {"success", "task_failure", "agent_timeout"}
)


def behavioral_task_output_available(row: Mapping[str, Any]) -> bool:
    """Return whether a typed terminal can own a behavioral task verdict."""

    terminal_kind = str(row.get("terminal_kind") or "")
    runtime_outcome = str(row.get("runtime_outcome") or "")
    if terminal_kind:
        if terminal_kind not in BEHAVIORAL_TERMINAL_KINDS:
            return False
        if runtime_outcome:
            return runtime_outcome in {"completed", "timed_out"}
        return (
            str(row.get("benchmark_outcome") or "") in {"passed", "failed"}
            or isinstance(row.get("pass"), bool)
        )
    if runtime_outcome:
        return runtime_outcome in {"completed", "timed_out"}
    return (
        str(row.get("benchmark_outcome") or "") in {"passed", "failed"}
        or str(row.get("status") or "") in {"passed", "failed"}
        or isinstance(row.get("pass"), bool)
    )


def nonbehavioral_score_field_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return fixed public score fields that cannot survive an unscored terminal."""

    candidates = (
        *DIRECT_SCORE_FIELDS,
        *(f"evaluation_{name}" for name in GENERATED_EVALUATION_SCORE_DIMENSIONS),
        *(
            f"prompt_injection_{name}"
            for name in (
                *PROMPT_INJECTION_REWARDS,
                *PROMPT_INJECTION_OPTIONAL_REWARDS,
            )
        ),
        "prompt_injection_classification",
        *EVIDENCE_USE_REWARDS,
    )
    return tuple(sorted(name for name in candidates if row.get(name) is not None))
