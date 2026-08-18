from __future__ import annotations

import pytest

from fugue.bench.harbor_outcome import classify_harbor_terminal


def _job(*, errored: int = 1, cancelled: int = 0) -> dict[str, object]:
    return {
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_errored_trials": errored,
            "n_cancelled_trials": cancelled,
            "evals": {},
        },
    }


def _trial(
    exception_type: str,
    *,
    agent_started: bool = False,
) -> dict[str, object]:
    return {
        "exception_info": {
            "exception_type": exception_type,
            "exception_message": "untrusted provider detail",
        },
        "agent_execution": (
            {"started_at": "2026-01-01T00:00:00Z"}
            if agent_started
            else None
        ),
        "environment_setup": {"started_at": "2026-01-01T00:00:00Z"},
        "agent_setup": None,
    }


def test_pre_agent_allowlisted_failure_is_typed_infrastructure() -> None:
    outcome = classify_harbor_terminal(
        _job(),
        cell_id="cell",
        trial=_trial("EnvironmentStartTimeoutError"),
    )

    assert outcome["terminal_kind"] == "sandbox_lost"
    assert outcome["status"] == "failed"
    assert outcome["runtime_outcome"] == "not_started"
    assert "untrusted provider detail" not in str(outcome["error"])


def test_bound_agent_transport_failure_is_typed_infrastructure() -> None:
    outcome = classify_harbor_terminal(
        _job(),
        cell_id="cell",
        trial=_trial("ApiConnectionClosedError", agent_started=True),
    )

    assert outcome["terminal_kind"] == "transport_interrupted"
    assert outcome["status"] == "failed"
    assert outcome["runtime_outcome"] == "interrupted"


@pytest.mark.parametrize(
    "exception_type",
    [
        "ApiInternalServerError",
        "ApiOverloadedError",
        "ApiRateLimitError",
        "ApiUsageLimitError",
    ],
)
def test_provider_outage_is_fatal_without_scoring_or_automatic_retry(
    exception_type: str,
) -> None:
    outcome = classify_harbor_terminal(
        _job(),
        cell_id="cell",
        trial=_trial(exception_type, agent_started=True),
    )

    assert outcome["terminal_kind"] == "execution_failure"
    assert outcome["status"] == "failed"
    assert outcome["runtime_outcome"] == "interrupted"


@pytest.mark.parametrize(
    "exception_type",
    [
        "DockerException",
        "ConnectError",
        "NetworkConnectionError",
        "UnknownApiError",
        "VerifierTimeoutError",
        "RuntimeError",
    ],
)
def test_started_or_unknown_failures_remain_behavioral(
    exception_type: str,
) -> None:
    outcome = classify_harbor_terminal(
        _job(),
        cell_id="cell",
        trial=_trial(exception_type, agent_started=True),
    )

    assert outcome["terminal_kind"] == "task_failure"
    assert outcome["runtime_outcome"] == "completed"


def test_unknown_pre_agent_failure_is_fatal_not_behavioral_or_retryable() -> None:
    outcome = classify_harbor_terminal(
        _job(),
        cell_id="cell",
        trial=_trial("RuntimeError"),
    )

    assert outcome["terminal_kind"] == "execution_failure"
    assert outcome["runtime_outcome"] == "not_started"


def test_agent_timeout_and_cancellation_remain_non_retryable_behavior() -> None:
    timeout = classify_harbor_terminal(
        _job(),
        cell_id="cell",
        trial=_trial("AgentTimeoutError", agent_started=True),
    )
    cancelled = classify_harbor_terminal(
        _job(cancelled=1),
        cell_id="cell",
        trial=_trial("CancelledError", agent_started=True),
    )

    assert timeout["terminal_kind"] == "agent_timeout"
    assert timeout["runtime_outcome"] == "timed_out"
    assert cancelled["terminal_kind"] == "cancelled"
    assert cancelled["runtime_outcome"] == "cancelled"


@pytest.mark.parametrize(
    ("job", "trial", "message"),
    [
        (_job(errored=0), _trial("DockerException"), "error count"),
        (_job(errored=1), {"exception_info": None}, "error count"),
        (_job(errored=1, cancelled=1), _trial("RuntimeError"), "cancellation"),
        (_job(errored=1), _trial("AgentTimeoutError"), "before Agent"),
    ],
)
def test_job_and_trial_contradictions_fail_closed(
    job: dict[str, object],
    trial: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        classify_harbor_terminal(job, cell_id="cell", trial=trial)
