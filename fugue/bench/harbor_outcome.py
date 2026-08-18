from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from fugue.bench.candidates import stable_digest

_SANDBOX_INTERRUPTION_TYPES = frozenset({"EnvironmentStartTimeoutError"})
_TRANSPORT_INTERRUPTION_TYPES = frozenset({"ApiConnectionClosedError"})
_PROVIDER_UNAVAILABLE_TYPES = frozenset(
    {
        "ApiInternalServerError",
        "ApiOverloadedError",
        "ApiRateLimitError",
        "ApiUsageLimitError",
    }
)
_PRE_AGENT_INFRASTRUCTURE_MARKERS = (
    "all predefined address pools have been fully subnetted",
    "cannot connect to the docker daemon",
)
HARBOR_TERMINAL_CLASSIFIER_DIGEST = stable_digest(
    {
        "schema_version": 1,
        "contract": "fugue.harbor-terminal-classifier",
        "sandbox_lost": sorted(_SANDBOX_INTERRUPTION_TYPES),
        "transport_interrupted": sorted(_TRANSPORT_INTERRUPTION_TYPES),
        "provider_unavailable_fatal": sorted(_PROVIDER_UNAVAILABLE_TYPES),
        "pre_agent_infrastructure_markers": list(
            _PRE_AGENT_INFRASTRUCTURE_MARKERS
        ),
        "agent_timeout": ["AgentTimeoutError"],
        "cancelled": ["CancelledError"],
        "unknown_pre_agent": "execution_failure",
        "unknown_post_agent": "task_failure",
    }
)


def classify_harbor_terminal(
    raw: Any,
    *,
    cell_id: str,
    trial: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one terminal Harbor trial from structured evidence only."""

    if not isinstance(raw, Mapping):
        raise ValueError("terminal Harbor JobResult must be an object")
    if int(raw.get("n_total_trials") or 0) != 1:
        raise ValueError("terminal Harbor JobResult must contain one trial")
    stats = raw.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError("terminal Harbor JobResult has no stats")
    completed = int(stats.get("n_completed_trials") or 0)
    running = int(stats.get("n_running_trials") or 0)
    pending = int(stats.get("n_pending_trials") or 0)
    if (completed, running, pending) != (1, 0, 0):
        raise ValueError("Harbor JobResult is not terminal")
    errored = int(stats.get("n_errored_trials") or 0)
    cancelled = int(stats.get("n_cancelled_trials") or 0)
    exception = trial.get("exception_info")
    exception_type = (
        str(exception.get("exception_type") or "")
        if isinstance(exception, Mapping)
        else ""
    )
    exception_message = (
        str(exception.get("exception_message") or "").lower()
        if isinstance(exception, Mapping)
        else ""
    )
    agent_started = _trial_agent_started(trial)
    environment_phase = _trial_environment_phase(trial)
    if cancelled not in {0, 1} or errored not in {0, 1}:
        raise ValueError("one Harbor cell has contradictory terminal counts")
    if bool(exception_type) != bool(errored):
        raise ValueError("Harbor error count disagrees with TrialResult exception")
    if (cancelled == 1) != (exception_type == "CancelledError"):
        raise ValueError("Harbor cancellation count disagrees with TrialResult")
    if exception_type == "AgentTimeoutError" and not agent_started:
        raise ValueError("Harbor Agent timeout occurred before Agent execution")
    rewards = _job_rewards(stats)
    reward = rewards[0] if rewards else None
    if cancelled or exception_type == "CancelledError":
        status = "cancelled"
        runtime = "cancelled"
        terminal_kind = "cancelled"
        error = "Harbor trial was cancelled"
        benchmark = "unscored"
    elif exception_type == "AgentTimeoutError":
        status = "failed"
        runtime = "timed_out"
        terminal_kind = "agent_timeout"
        error = "Harbor Agent execution timed out"
        benchmark = "unscored"
    elif (
        environment_phase
        and not agent_started
        and exception_type in _SANDBOX_INTERRUPTION_TYPES
    ):
        status = "failed"
        runtime = "not_started"
        terminal_kind = "sandbox_lost"
        error = f"Harbor sandbox infrastructure failed: {exception_type}"
        benchmark = "unscored"
    elif (
        environment_phase
        and not agent_started
        and exception_type == "RuntimeError"
        and any(
            marker in exception_message
            for marker in _PRE_AGENT_INFRASTRUCTURE_MARKERS
        )
    ):
        status = "failed"
        runtime = "not_started"
        terminal_kind = "sandbox_lost"
        error = "Harbor Docker network infrastructure was unavailable"
        benchmark = "unscored"
    elif agent_started and exception_type in _TRANSPORT_INTERRUPTION_TYPES:
        status = "failed"
        runtime = "interrupted"
        terminal_kind = "transport_interrupted"
        error = f"Harbor Agent transport was interrupted: {exception_type}"
        benchmark = "unscored"
    elif agent_started and exception_type in _PROVIDER_UNAVAILABLE_TYPES:
        # These are not task outcomes, but automatically replacing a paid
        # Agent execution can duplicate spend. Halt the governed run for an
        # operator decision instead of scoring or retrying it.
        status = "failed"
        runtime = "interrupted"
        terminal_kind = "execution_failure"
        error = f"Harbor model provider was unavailable: {exception_type}"
        benchmark = "unscored"
    elif environment_phase and not agent_started and exception_type:
        status = "failed"
        runtime = "not_started"
        terminal_kind = "execution_failure"
        error = (
            "Harbor failed before Agent execution without a retryable "
            f"infrastructure classification: {exception_type}"
        )
        benchmark = "unscored"
    elif errored:
        status = "failed"
        runtime = "completed"
        terminal_kind = "task_failure"
        error = f"{errored} Harbor trial(s) errored"
        benchmark = "unscored"
    else:
        status = "passed"
        runtime = "completed"
        terminal_kind = "success"
        error = None
        benchmark = (
            "passed" if reward == 1.0 else "failed" if reward is not None else "unscored"
        )
    return {
        "cell_id": cell_id,
        "status": status,
        "returncode": None,
        "error": error,
        "benchmark_outcome": benchmark,
        "reward": reward,
        "runtime_outcome": runtime,
        "terminal_kind": terminal_kind,
    }


def _job_rewards(stats: Mapping[str, Any]) -> list[float]:
    rewards: list[float] = []
    for evaluation in (stats.get("evals") or {}).values():
        if not isinstance(evaluation, Mapping):
            continue
        buckets = ((evaluation.get("reward_stats") or {}).get("reward") or {})
        if not isinstance(buckets, Mapping):
            continue
        for raw_reward, trial_ids in buckets.items():
            reward = float(raw_reward)
            if not math.isfinite(reward):
                raise ValueError("Harbor reward is not finite")
            count = len(trial_ids) if isinstance(trial_ids, list) else 1
            rewards.extend([reward] * count)
    if len(rewards) > 1:
        raise ValueError("one Harbor cell produced multiple rewards")
    return rewards


def _trial_agent_started(trial: Mapping[str, Any]) -> bool:
    if trial.get("agent_execution") is not None:
        return True
    steps = trial.get("step_results")
    return bool(
        isinstance(steps, list)
        and any(
            isinstance(step, Mapping)
            and (
                step.get("agent_execution") is not None
                or step.get("agent_result") is not None
            )
            for step in steps
        )
    )


def _trial_environment_phase(trial: Mapping[str, Any]) -> bool:
    environment = trial.get("environment_setup")
    return bool(
        isinstance(environment, Mapping)
        and environment.get("started_at")
        and trial.get("agent_setup") is None
    )
