from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.task_interaction import TaskInteractionController


def _metadata(controller: dict[str, object]) -> str:
    return json.dumps(
        {
            "task_definition_digest": "a" * 64,
            "criteria_digest": "b" * 64,
            "interaction_controller": controller,
        }
    )


def test_scripted_interaction_reveals_turns_only_when_they_are_due(
    tmp_path: Path,
) -> None:
    env = {
        "FUGUE_TRACE_CONTENT": "full",
        "FUGUE_TASK_AUTHORING": _metadata(
            {
                "type": "scripted",
                "profile_id": "scripted-v1",
                "profile_digest": "c" * 64,
                "scripted_turns": ["Show the evidence.", "Now summarize the risk."],
                "directions": [],
                "max_user_turns": 2,
                "max_agent_turns": 3,
                "timeout_sec": 60,
            }
        ),
    }
    controller = TaskInteractionController.from_environment(
        logs_dir=tmp_path,
        initial_instruction="Investigate the failure.",
        env=env,
    )

    assert not (tmp_path / "task-interaction.jsonl").exists()
    controller.observe_agent("The first hypothesis is a cache mismatch.")
    first = controller.next_follow_up(0)
    evidence = (tmp_path / "task-interaction.jsonl").read_text()
    assert first == "Show the evidence."
    assert "Now summarize the risk." not in evidence

    controller.observe_agent("The trace and cache key disagree.")
    second = controller.next_follow_up(1)
    controller.observe_agent("The risk is stale evidence reuse.")
    assert second == "Now summarize the risk."
    assert controller.summary()["observed_agent_turns"] == 3
    assert controller.summary()["planned_agent_turns"] == 3


def test_metadata_trace_records_hashes_without_conversation_content(
    tmp_path: Path,
) -> None:
    env = {
        "FUGUE_TRACE_CONTENT": "metadata",
        "FUGUE_TASK_AUTHORING": _metadata(
            {
                "type": "scripted",
                "profile_id": "scripted-v1",
                "profile_digest": "c" * 64,
                "scripted_turns": ["Private follow-up text."],
                "directions": [],
                "max_user_turns": 1,
                "max_agent_turns": 2,
                "timeout_sec": 60,
            }
        ),
    }
    controller = TaskInteractionController.from_environment(
        logs_dir=tmp_path,
        initial_instruction="Private task text.",
        env=env,
    )
    controller.observe_agent("Private response text.")
    controller.next_follow_up(0)

    evidence = (tmp_path / "task-interaction.jsonl").read_text()
    assert "Private response text." not in evidence
    assert "Private follow-up text." not in evidence
    assert "content_sha256" in evidence


def test_model_interactor_has_a_separate_route_receipt_and_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {
        "OPENAI_API_KEY": "model-secret",
        "FUGUE_TRACE_CONTENT": "full",
        "FUGUE_TASK_AUTHORING": _metadata(
            {
                "type": "model",
                "profile_id": "clarifier-v1",
                "profile_digest": "d" * 64,
                "model": "openai/gpt-5",
                "scripted_turns": [],
                "directions": ["Ask for one missing piece of evidence."],
                "max_user_turns": 1,
                "max_agent_turns": 2,
                "timeout_sec": 60,
                "input_cost_per_million": 10,
                "output_cost_per_million": 20,
            }
        ),
    }

    def fake_post(*args, **kwargs):
        del args, kwargs
        return {"follow_up": "Which trace supports that?"}, {
            "input_tokens": 10,
            "output_tokens": 5,
        }

    monkeypatch.setattr("fugue.task_interaction._post_judge", fake_post)
    controller = TaskInteractionController.from_environment(
        logs_dir=tmp_path,
        initial_instruction="Diagnose the issue.",
        env=env,
    )
    controller.observe_agent("It appears to be a race.")

    assert controller.next_follow_up(0) == "Which trace supports that?"
    [receipt] = [
        json.loads(line)
        for line in (tmp_path / "interactor-route-receipts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert receipt["role"] == "interactor"
    assert receipt["trace_scope"] == "separate_from_agent"
    assert receipt["route"]["display_model"] == "openai/gpt-5"
    assert receipt["cost_usd"] == pytest.approx(0.0002)
    assert controller.summary()["observed_interactor_cost_usd"] == pytest.approx(0.0002)
    serialized = json.dumps(receipt, sort_keys=True)
    assert "model-secret" not in serialized
    assert "criteria" not in serialized


def test_every_harness_uses_native_session_continuation() -> None:
    source = Path("fugue/agents/model_plane.py").read_text()

    assert "hermes --yolo chat --continue" in source
    assert "--session-key {shlex.quote(session_key)}" in source
    assert "claude --continue --verbose" in source
    assert "codex exec resume --last" in source
