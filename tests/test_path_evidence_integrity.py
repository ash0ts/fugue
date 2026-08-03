from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from fugue.agents.model_plane import _TrialMetaMixin
from fugue.bench import export


class _TrialStub(_TrialMetaMixin):
    TRACE_HARNESS = "claude-code"

    def __init__(self, logs_dir: Path, *, collection_error: Exception | None = None):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True)
        self._collection_error = collection_error
        self._runtime_fingerprints = {}
        self._fugue_secret_files = {}

    async def _restore_verifier_runtime(self, environment: object) -> None:
        return None

    async def _capture_runtime_fingerprint(
        self, environment: object, stage: str
    ) -> None:
        return None

    async def _normalize_artifact_paths(
        self, environment: object
    ) -> list[dict[str, str]]:
        return []

    async def _container_repo_root(self, environment: object) -> str:
        if self._collection_error is not None:
            raise self._collection_error
        return "/workspace/repo"

    async def exec_as_agent(
        self,
        environment: object,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            return_code=0,
            stdout="src/app.py\ntests/test_app.py\nsrc/app.py\n",
            stderr="",
        )

    async def _remove_exec_secrets(self, environment: object) -> None:
        return None

    def _extract_session_ids(self) -> list[str]:
        return []


def test_post_run_git_diff_records_available_empty_or_nonempty_receipt(
    tmp_path: Path,
) -> None:
    trial = _TrialStub(tmp_path / "agent")

    asyncio.run(trial._finish_trial(object()))

    meta = json.loads((trial.logs_dir / "fugue-meta.json").read_text())
    assert meta["changed_paths_status"] == "available"
    assert meta["changed_paths"] == ["src/app.py", "tests/test_app.py"]
    assert "changed_paths_error" not in meta


def test_post_run_git_diff_failure_is_unavailable_not_verified_empty(
    tmp_path: Path,
) -> None:
    trial = _TrialStub(
        tmp_path / "agent",
        collection_error=RuntimeError("sensitive failure detail"),
    )

    asyncio.run(trial._finish_trial(object()))

    meta = json.loads((trial.logs_dir / "fugue-meta.json").read_text())
    assert meta["changed_paths_status"] == "unavailable"
    assert meta["changed_paths"] == []
    assert meta["changed_paths_error"] == "RuntimeError"
    assert "sensitive failure detail" not in json.dumps(meta)


def test_missing_or_malformed_trajectory_marks_inspection_unavailable(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial"

    missing = export._trajectory_activity(trial)
    assert missing["inspected_paths_status"] == "unavailable"
    assert missing["inspected_paths"] == []

    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "trajectory.json").write_text("[]")
    malformed = export._trajectory_activity(trial)
    assert malformed["inspected_paths_status"] == "unavailable"
    assert malformed["inspected_paths"] == []


def test_trajectory_counts_only_correlated_successful_path_activity(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "tool_calls": [
                            {
                                "tool_call_id": "successful-read",
                                "function_name": "read_file",
                                "arguments": {"path": "src/read.py"},
                            },
                            {
                                "tool_call_id": "failed-read",
                                "function_name": "read_file",
                                "arguments": {"path": "src/not-read.py"},
                            },
                            {
                                "tool_call_id": "successful-write",
                                "function_name": "write_file",
                                "arguments": {"path": "src/written.py"},
                            },
                            {
                                "tool_call_id": "unresolved-write",
                                "function_name": "write_file",
                                "arguments": {"path": "src/not-written.py"},
                            },
                        ],
                        "observation": {
                            "results": [
                                {
                                    "source_call_id": "successful-read",
                                    "content": "source",
                                },
                                {
                                    "source_call_id": "failed-read",
                                    "content": "permission denied",
                                    "extra": {"tool_result_is_error": True},
                                },
                                {
                                    "source_call_id": "successful-write",
                                    "content": "ok",
                                },
                            ]
                        },
                    }
                ]
            }
        )
    )

    activity = export._trajectory_activity(trial)

    assert activity["inspected_paths_status"] == "available"
    assert activity["inspected_paths"] == ["src/read.py"]
    assert activity["changed_paths"] == ["src/written.py"]
    assert len(activity["error_events"]) == 1
    assert activity["error_events"][0]["tool_name"] == "read_file"
    assert activity["error_events"][0]["source"] == "local_trajectory"
