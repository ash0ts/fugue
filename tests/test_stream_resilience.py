from __future__ import annotations

import errno
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench import cli
from fugue.bench.execution import PlannedCell, execute_cells
from fugue.bench.streams import write_stdout_best_effort


class _ClosedConsumer:
    closed = False

    def write(self, _text: str) -> int:
        raise BrokenPipeError(errno.EPIPE, "stdout consumer closed")

    def flush(self) -> None:
        raise AssertionError("flush must not run after a failed write")


class _UnexpectedFailure:
    closed = False

    def write(self, _text: str) -> int:
        raise OSError(errno.ENOSPC, "unexpected output failure")

    def flush(self) -> None:
        raise AssertionError("flush must not run after a failed write")


def _diagnostic_cell(tmp_path: Path, *, exit_code: int) -> PlannedCell:
    run_id = f"run-closed-stdout-{exit_code}"
    return PlannedCell(
        id=f"cell-closed-stdout-{exit_code}",
        run_id=run_id,
        run_name=run_id,
        workload_id="diagnostic",
        task_id="closed-stdout",
        harness="direct",
        context_system_id="none",
        variant_id="baseline",
        model_provider="none",
        model="none",
        trial_index=1,
        comparison_example_id="closed-stdout",
        candidate_id="diagnostic",
        execution_fingerprint="runtime",
        config_path=tmp_path / "unused.json",
        result_path=tmp_path / "unused-result.json",
        command=(
            sys.executable,
            "-c",
            f"print('durable child output'); raise SystemExit({exit_code})",
        ),
        env={},
        n_attempts=1,
        execution_kind="provider_diagnostic",
    )


def test_closed_consumer_is_distinct_from_unexpected_stdout_failure() -> None:
    assert write_stdout_best_effort("output", stream=_ClosedConsumer()) is False
    with pytest.raises(OSError, match="unexpected output failure"):
        write_stdout_best_effort("output", stream=_UnexpectedFailure())


@pytest.mark.parametrize(
    ("exit_code", "expected_status"),
    ((0, "passed"), (7, "failed")),
)
def test_cell_log_remains_durable_when_stdout_consumer_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    expected_status: str,
) -> None:
    cell = _diagnostic_cell(tmp_path, exit_code=exit_code)
    monkeypatch.setattr(
        "fugue.bench.execution.write_stdout_best_effort",
        lambda text: write_stdout_best_effort(text, stream=_ClosedConsumer()),
    )

    [outcome] = execute_cells([cell], repo_root=tmp_path, max_workers=1)

    assert outcome.status == expected_status
    assert outcome.returncode == exit_code
    assert "cell log reader failed" not in (outcome.error or "")
    runtime = tmp_path / ".fugue" / "runtime" / cell.run_id
    assert "durable child output" in (runtime / "logs" / f"{cell.id}.log").read_text()
    events = [
        json.loads(line)
        for line in (runtime / "events.jsonl").read_text().splitlines()
    ]
    assert any(event["event"] == "log" for event in events)
    detached = [
        event for event in events if event["event"] == "cell_log_stream_detached"
    ]
    assert len(detached) == 1
    assert detached[0]["sink"] == "stdout"


def test_wait_for_run_detaches_observer_without_cancelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled: list[str] = []

    class _Supervisor:
        def follow_log(self, _run_id: str):
            yield "durable worker output\n"

        def cancel(self, run_id: str) -> None:
            cancelled.append(run_id)

    service = SimpleNamespace(supervisor=_Supervisor())
    monkeypatch.setattr(cli, "CONSOLE", SimpleNamespace(is_terminal=False))
    monkeypatch.setattr(cli, "write_stdout_best_effort", lambda _chunk: False)

    assert cli._wait_for_run(service, "run-detached") == 0
    assert cancelled == []
