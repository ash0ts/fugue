from __future__ import annotations

import asyncio
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fugue.bench.campaign_lifecycle as campaign_lifecycle
import fugue.bench.job_config as job_config
from fugue.agents import FugueWBAResponses
from fugue.bench.campaign_evidence import safe_prediction_row
from fugue.bench.campaigns import CampaignService, build_experiment_proposal
from fugue.bench.export import _wba_transport_evidence
from fugue.bench.manifest import load_manifest
from fugue.bench.operator import ExperimentRequest, OperatorService
from fugue.bench.wba_transport_analysis import analyze_wba_transport_rows
from fugue.bench.wba_transport_tasks import WBATransportTaskMaterializer
from fugue.task_interaction import TaskInteractionController

RUNNER_PATH = Path("configs/fugue/runtime/wba-responses/wba-runner")
RUNNER = runpy.run_path(RUNNER_PATH.as_posix())


class _Events:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, **values: Any) -> None:
        self.rows.append((event, values))


class _Trace:
    def start_llm(self, kind: str, groups: list[list[dict[str, Any]]]) -> None:
        del kind, groups
        return None

    def finish_llm(self, span: Any, result: Any, error: Any) -> None:
        del span, result, error

    def start_tool(self, call: Any) -> None:
        del call
        return None

    def finish_tool(self, span: Any, output: str, error: Any) -> None:
        del span, output, error


def test_wba_weave_trace_records_assistant_output_as_a_message() -> None:
    class _Turn:
        def __init__(self) -> None:
            self.messages: list[Any] = []
            self.exited = False

        def __exit__(self, *args: Any) -> None:
            del args
            self.exited = True

    trace = RUNNER["WeaveTrace"]({"trace_content": "full"})
    turn = _Turn()
    trace.turn = turn

    trace.finish("final answer")

    assert turn.exited is True
    assert len(turn.messages) == 1
    assert turn.messages[0].role == "assistant"
    assert turn.messages[0].content == "final answer"


def test_wba_weave_trace_disables_implicit_sdk_patching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weave

    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    captured: dict[str, Any] = {}

    def fake_init(project: str, **kwargs: Any) -> None:
        captured.update(project=project, **kwargs)
        raise RuntimeError("stop after settings capture")

    monkeypatch.setattr(weave, "init", fake_init)
    trace = RUNNER["WeaveTrace"](
        {
            "trace_content": "full",
            "weave_project": "wandb/test",
            "conversation_id": "conversation-1",
            "display_model": "wandb/model",
        }
    )

    trace.start("instruction")

    assert captured == {
        "project": "wandb/test",
        "settings": {"implicitly_patch_integrations": False},
    }
    assert trace.conversation is None
    assert trace.turn is None


def _client(profile: str) -> Any:
    return RUNNER["ModelClient"](
        {
            "profile": profile,
            "model_id": "zai-org/GLM-5.2",
            "litellm_model": "nebius/zai-org/GLM-5.2",
            "provider_key_env": "WANDB_API_KEY",
            "provider_base_url": "https://api.inference.wandb.ai/v1",
            "provider_headers": {"OpenAI-Project": "wandb/fugue-test"},
            "system_prompt": "fixed",
        },
        _Events(),
        _Trace(),
    )


def test_wba_adapter_rejects_unknown_transport_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_ENTITY", "wandb")
    monkeypatch.setenv("WANDB_PROJECT", "fugue-test")

    with pytest.raises(ValueError, match="unsupported WBA transport profile"):
        FugueWBAResponses(
            logs_dir=tmp_path,
            model_name="wandb/zai-org/GLM-5.2",
            transport_profile="arbitrary-provider",
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"provider": "unregistered"},
        {"base_url": "https://arbitrary.example/v1"},
        {"dependency_override": "litellm==latest"},
    ],
)
def test_wba_adapter_rejects_unregistered_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: dict[str, str],
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_ENTITY", "wandb")
    monkeypatch.setenv("WANDB_PROJECT", "fugue-test")

    with pytest.raises(ValueError, match="unknown WBA agent configuration"):
        FugueWBAResponses(
            logs_dir=tmp_path,
            model_name="wandb/zai-org/GLM-5.2",
            **extra,
        )


def test_wba_adapter_explicitly_rejects_native_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_ENTITY", "wandb")
    monkeypatch.setenv("WANDB_PROJECT", "fugue-test")

    with pytest.raises(ValueError, match="does not support native MCP"):
        FugueWBAResponses(
            logs_dir=tmp_path,
            model_name="wandb/zai-org/GLM-5.2",
            mcp_servers=[SimpleNamespace(name="unsafe")],
        )


def test_wba_adapter_exposes_the_locked_transport_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_ENTITY", "wandb")
    monkeypatch.setenv("WANDB_PROJECT", "fugue-test")
    agent = FugueWBAResponses(
        logs_dir=tmp_path,
        model_name="wandb/zai-org/GLM-5.2",
        transport_profile="chat-inline",
    )

    assert agent.transport_receipt["profile"] == "chat-inline"
    assert agent.transport_receipt["bridge_required"] is False
    assert agent.transport_receipt["codec"] == "chat-completions-native-v1"
    assert len(str(agent.transport_receipt["retry_policy_digest"])) == 64
    assert len(str(agent.transport_receipt["timeout_policy_digest"])) == 64
    assert len(str(agent.transport_receipt["compaction_policy_digest"])) == 64


def test_wba_adapter_executes_through_the_harbor_agent_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_ENTITY", "wandb")
    monkeypatch.setenv("WANDB_PROJECT", "fugue-test")
    agent = FugueWBAResponses(
        logs_dir=tmp_path,
        model_name="wandb/zai-org/GLM-5.2",
        transport_profile="chat-inline",
    )

    class Environment:
        default_user = "agent-user"

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def exec(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    environment = Environment()
    asyncio.run(
        agent.exec_as_agent(
            environment,
            command="pwd",
            env={"SAFE_VALUE": "one"},
            timeout_sec=10,
        )
    )

    assert environment.calls == [
        {
            "command": "pwd",
            "env": {"SAFE_VALUE": "one"},
            "cwd": None,
            "timeout_sec": 10,
            "user": "agent-user",
        }
    ]


def test_chat_stream_reconciles_fragmented_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def completion(**kwargs: Any) -> Any:
        del kwargs

        async def chunks():
            yield {
                "id": "response-a",
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "inspect ",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_",
                                    "function": {
                                        "name": "she",
                                        "arguments": '{"command":"py',
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "evidence",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "1",
                                    "function": {
                                        "name": "ll",
                                        "arguments": 'thon -V"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            }

        return chunks()

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=completion))
    monkeypatch.setenv("WANDB_API_KEY", "test-only")

    async def exercise() -> Any:
        client = _client("chat-inline")
        try:
            return await client._chat_inline([], stream=True)
        finally:
            await client.close()

    result = asyncio.run(exercise())

    assert result.reasoning == {"text": "inspect evidence"}
    assert result.tool_calls[0].call_id == "call_1"
    assert result.tool_calls[0].name == "shell"
    assert result.tool_calls[0].arguments == {"command": "python -V"}
    assert result.input_tokens == 12
    assert result.output_tokens == 8


def test_inline_profiles_pass_an_owned_openai_client_to_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-only")

    async def exercise() -> tuple[Any, Any, bool]:
        client = _client("chat-inline")
        common = client._common()
        inline = client._inline_openai
        try:
            return common["client"], inline, hasattr(inline, "chat")
        finally:
            await client.close()

    passed, owned, supports_chat = asyncio.run(exercise())

    assert passed is owned
    assert supports_chat is True


def test_litellm_remote_cost_map_is_disabled_in_locked_runtime() -> None:
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "true"


def test_responses_parser_preserves_reasoning_and_multiple_call_ids() -> None:
    result = RUNNER["_parse_responses"](
        {
            "id": "response-a",
            "status": "completed",
            "usage": {"input_tokens": 5, "output_tokens": 7},
            "output": [
                {"type": "reasoning", "summary": [{"text": "inspect"}]},
                {
                    "type": "function_call",
                    "call_id": "call-a",
                    "name": "shell",
                    "arguments": '{"command":"ls"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call-b",
                    "name": "shell",
                    "arguments": '{"command":"pwd"}',
                },
            ],
        }
    )

    assert result.reasoning["type"] == "reasoning"
    assert [call.call_id for call in result.tool_calls] == ["call-a", "call-b"]
    assert result.finish_reason == "completed"


def test_responses_parser_preserves_text_only_output() -> None:
    result = RUNNER["_parse_responses"](
        {
            "id": "response-text",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
        }
    )

    assert result.text == "done"
    assert result.tool_calls == []


def test_weave_chat_payload_preserves_visible_history_without_reasoning() -> None:
    messages = RUNNER["_trace_messages"](
        [
            [{"role": "user", "content": "inspect the fixture"}],
            [
                {"type": "reasoning", "summary": [{"text": "hidden"}]},
                {
                    "type": "function_call",
                    "call_id": "call-a",
                    "name": "shell",
                    "arguments": '{"command":"pwd"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-a",
                    "output": "/workspace",
                },
            ],
        ]
    )

    assert "hidden" not in json.dumps(messages)
    assert messages[0] == {"role": "user", "content": "inspect the fixture"}
    assert messages[1]["tool_calls"][0]["id"] == "call-a"
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call-a",
        "content": "/workspace",
    }


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"id": "", "name": "shell", "arguments": "{}"}, "durable call id"),
        (
            {"id": "call-a", "name": "shell", "arguments": "{bad"},
            "malformed tool arguments",
        ),
    ],
)
def test_tool_call_parser_fails_closed(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RUNNER["_tool_call"](raw)


def test_history_builder_rejects_missing_tool_outputs() -> None:
    turn = RUNNER["ModelTurn"](
        tool_calls=[RUNNER["ToolCall"]("call-a", "shell", {"command": "pwd"})]
    )

    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        RUNNER["_assistant_group"]("chat-inline", turn, [])


def test_shell_tool_error_is_recoverable_and_bounded(tmp_path: Path) -> None:
    events = _Events()
    call = RUNNER["ToolCall"](
        "call-a",
        "shell",
        {"command": "printf failure >&2; exit 7"},
    )

    output, is_error = asyncio.run(
        RUNNER["_execute_tool"](
            call,
            {"workspace": tmp_path.as_posix()},
            events,
            _Trace(),
        )
    )

    assert is_error is True
    assert "status 7" in output
    assert "failure" in output
    assert events.rows[-1][1]["call_id"] == "call-a"


def test_shell_tool_cancellation_kills_the_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 4242
        returncode = -9

        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()

        async def communicate(self) -> tuple[bytes, bytes]:
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await asyncio.Future()
            return b"", b"cancelled"

    process = Process()
    signals: list[tuple[int, int]] = []

    async def create_subprocess(*args: Any, **kwargs: Any) -> Process:
        del args, kwargs
        return process

    monkeypatch.setattr(RUNNER["asyncio"], "create_subprocess_shell", create_subprocess)
    monkeypatch.setattr(
        RUNNER["os"],
        "killpg",
        lambda pid, signal_number: signals.append((pid, signal_number)),
    )

    async def exercise() -> None:
        task = asyncio.create_task(
            RUNNER["_execute_tool"](
                RUNNER["ToolCall"]("call-a", "shell", {"command": "sleep 60"}),
                {"workspace": tmp_path.as_posix()},
                _Events(),
                _Trace(),
            )
        )
        await process.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert signals == [(4242, RUNNER["signal"].SIGKILL)]


def test_shell_tool_timeout_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 4343
        returncode = -9

        def __init__(self) -> None:
            self.calls = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError
            return b"partial output", b""

    process = Process()
    signals: list[tuple[int, int]] = []

    async def create_subprocess(*args: Any, **kwargs: Any) -> Process:
        del args, kwargs
        return process

    monkeypatch.setattr(RUNNER["asyncio"], "create_subprocess_shell", create_subprocess)
    monkeypatch.setattr(
        RUNNER["os"],
        "killpg",
        lambda pid, signal_number: signals.append((pid, signal_number)),
    )

    output, is_error = asyncio.run(
        RUNNER["_execute_tool"](
            RUNNER["ToolCall"]("call-a", "shell", {"command": "slow"}),
            {"workspace": tmp_path.as_posix()},
            _Events(),
            _Trace(),
        )
    )

    assert is_error is True
    assert "timed out" in output
    assert signals == [(4343, RUNNER["signal"].SIGKILL)]


def test_retry_policy_records_recoverable_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client("chat-inline")
    attempts = 0

    async def operation() -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary")
        return RUNNER["ModelTurn"](text="recovered")

    async def no_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(RUNNER["asyncio"], "sleep", no_sleep)
    result = asyncio.run(client._retry(operation, "agent"))

    assert result.text == "recovered"
    assert client.retries == 1
    assert client.transport_errors == 1
    assert client.normalization_errors == 0
    assert [event for event, _ in client.events.rows] == [
        "model_error",
        "model_retry",
    ]
    assert client.events.rows[0][1]["will_retry"] is True


def test_responses_stream_fails_closed_when_completion_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyIterator:
        def __aiter__(self) -> EmptyIterator:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    async def responses(**kwargs: Any) -> EmptyIterator:
        assert kwargs["stream"] is True
        return EmptyIterator()

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(aresponses=responses))
    monkeypatch.setenv("WANDB_API_KEY", "test-only")

    async def exercise() -> tuple[int, int]:
        client = _client("responses-inline")
        try:
            with pytest.raises(RuntimeError, match="ended without output"):
                await client.stream([])
            return client.transport_errors, client.normalization_errors
        finally:
            await client.close()

    assert asyncio.run(exercise()) == (1, 1)


def test_chat_stream_fails_closed_when_text_and_tools_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyIterator:
        def __aiter__(self) -> EmptyIterator:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    async def completion(**kwargs: Any) -> EmptyIterator:
        assert kwargs["stream"] is True
        return EmptyIterator()

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=completion))
    monkeypatch.setenv("WANDB_API_KEY", "test-only")

    async def exercise() -> tuple[int, int]:
        client = _client("chat-inline")
        try:
            with pytest.raises(RuntimeError, match="no text or tool calls"):
                await client.stream([])
            return client.transport_errors, client.normalization_errors
        finally:
            await client.close()

    assert asyncio.run(exercise()) == (1, 1)


def test_model_turn_rejects_duplicate_tool_call_ids() -> None:
    duplicate = RUNNER["ModelTurn"](
        tool_calls=[
            RUNNER["ToolCall"]("call-1", "shell", {"command": "pwd"}),
            RUNNER["ToolCall"]("call-1", "shell", {"command": "ls"}),
        ]
    )

    with pytest.raises(ValueError, match="duplicate tool-call IDs"):
        RUNNER["_validate_model_turn"](duplicate)


def test_failed_session_writes_terminal_transport_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def malformed_stream(client: Any, groups: Any) -> Any:
        del client, groups
        raise RuntimeError("Responses stream ended without output")

    monkeypatch.setattr(RUNNER["ModelClient"], "_stream_once", malformed_stream)
    monkeypatch.setenv("FUGUE_DISABLE_WEAVE", "1")
    events_path = tmp_path / "events.jsonl"
    summary_path = tmp_path / "summary.json"
    config = {
        "profile": "responses-inline",
        "session_id": "session-failed",
        "turns": ["Inspect the locked fixture."],
        "events_path": events_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "workspace": tmp_path.as_posix(),
        "system_prompt": "fixed",
        "model_id": "zai-org/GLM-5.2",
        "litellm_model": "nebius/zai-org/GLM-5.2",
        "provider_key_env": "WANDB_API_KEY",
        "provider_base_url": "https://api.inference.wandb.ai/v1",
        "provider_headers": {},
    }

    with pytest.raises(RuntimeError, match="ended without output"):
        asyncio.run(RUNNER["run"](config))

    summary = json.loads(summary_path.read_text())
    assert summary["event"] == "session_error"
    assert summary["stop_reason"] == "transport_error"
    assert summary["transport_errors"] == 1
    assert summary["normalization_errors"] == 1
    event_rows = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert event_rows[-1]["event"] == "session_summary"
    assert event_rows[-1]["stop_reason"] == "transport_error"


def test_compaction_preserves_head_and_tail_groups() -> None:
    class Client:
        profile = "chat-inline"

        async def summarize(self, text: str) -> str:
            assert "middle" in text
            return "locked summary"

    groups = [[{"role": "user", "content": f"middle-{index}"}] for index in range(7)]
    compacted, changed = asyncio.run(
        RUNNER["_compact_if_needed"](
            groups,
            Client(),
            {"system_prompt": "fixed", "context_window": 20},
            _Events(),
        )
    )

    assert changed is True
    assert compacted[0] == groups[0]
    assert compacted[-3:] == groups[-3:]
    assert "locked summary" in compacted[1][0]["content"]


def test_compaction_falls_back_without_losing_the_locked_history() -> None:
    class Client:
        profile = "responses-inline"

        async def summarize(self, text: str) -> str:
            del text
            raise ConnectionError("summary unavailable")

    events = _Events()
    groups = [[{"role": "user", "content": f"middle-{index}"}] for index in range(7)]

    compacted, changed = asyncio.run(
        RUNNER["_compact_if_needed"](
            groups,
            Client(),
            {"system_prompt": "fixed", "context_window": 20},
            events,
        )
    )

    assert changed is True
    assert compacted[0] == groups[0]
    assert compacted[-3:] == groups[-3:]
    assert "inspect the current workspace" in json.dumps(compacted[1])
    assert any(event == "compaction_fallback" for event, _value in events.rows)


def test_runner_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"profile": "chat-inline", "arbitrary_url": "x"}))

    with pytest.raises(ValueError, match="unknown runner config"):
        RUNNER["_load_config"](path)


def test_openai_client_lifecycle_closes_once() -> None:
    class Client:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    model_client = _client("responses-proxy")
    openai_client = Client()
    model_client._openai = openai_client

    asyncio.run(model_client.close())

    assert openai_client.closed == 1
    assert model_client._openai is None


def test_proxy_codec_normalizes_litellm_reasoning_index_reuse() -> None:
    completed = {
        "id": "resp-mock",
        "status": "completed",
        "usage": {"input_tokens": 9, "output_tokens": 5},
        "output": [
            {"id": "reason-1", "type": "reasoning", "summary": []},
            {
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "mock ok"}],
            },
            {
                "id": "call-1",
                "call_id": "call-1",
                "type": "function_call",
                "name": "shell",
                "arguments": '{"command":"pwd"}',
            },
        ],
    }
    events = [
        {"type": "response.created", "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {"id": "reason-1", "type": "reasoning"},
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 2,
            "output_index": 0,
            "item": {"id": "reason-1", "type": "reasoning"},
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 3,
            "output_index": 1,
            "item": {
                "id": "call-1",
                "call_id": "call-1",
                "type": "function_call",
                "name": "shell",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 4,
            "output_index": 1,
            "delta": '{"command":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 5,
            "output_index": 1,
            "delta": '"pwd"}',
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 6,
            "output_index": 1,
            "item": completed["output"][2],
        },
        {
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": "mock ok",
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": completed["output"][1],
        },
        {"type": "response.completed", "response": completed},
    ]

    class RawResponse:
        async def iter_lines(self) -> Any:
            for event in events:
                yield "data: " + json.dumps(event)
                yield ""
            yield "data: [DONE]"
            yield ""

    class Context:
        async def __aenter__(self) -> RawResponse:
            return RawResponse()

        async def __aexit__(self, *args: Any) -> None:
            del args

    def create(**kwargs: Any) -> Context:
        assert kwargs["stream"] is True
        return Context()

    proxy = SimpleNamespace(
        responses=SimpleNamespace(
            with_streaming_response=SimpleNamespace(create=create)
        )
    )
    model_client = _client("responses-proxy")
    model_client._openai = proxy

    result = asyncio.run(
        model_client._responses_proxy(
            [[{"role": "user", "content": "use shell"}]],
            stream=True,
        )
    )

    assert result.text == "mock ok"
    assert result.input_tokens == 9
    assert result.output_tokens == 5
    assert [
        (call.call_id, call.name, call.arguments) for call in result.tool_calls
    ] == [("call-1", "shell", {"command": "pwd"})]
    assert model_client.stream_events == len(events)
    assert model_client.normalization_errors == 0
    assert model_client.stream_anomalies >= 4
    assert sum(model_client.stream_anomaly_kinds.values()) == (
        model_client.stream_anomalies
    )
    assert any(
        event == "stream_normalized"
        and value["codec"].endswith("-v3")
        and value["stream_anomalies"] >= 4
        and value["anomaly_kinds"]
        for event, value in model_client.events.rows
    )


def test_proxy_codec_rejects_malformed_sse_json() -> None:
    with pytest.raises(ValueError, match="malformed SSE JSON"):
        RUNNER["_append_sse_event"]([], ["not-json"])


def test_inline_client_lifecycle_closes_once() -> None:
    class Client:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    model_client = _client("responses-inline")
    inline_client = Client()
    model_client._inline_openai = inline_client

    asyncio.run(model_client.close())

    assert inline_client.closed == 1
    assert model_client._inline_openai is None


def test_wba_registered_experiment_resolves_exact_locked_matrices() -> None:
    service = OperatorService(Path(__file__).parents[1])

    canary = service.resolve_run_plan(
        ExperimentRequest(
            experiment_id="wba-transport-ablation-v1",
            preset="canary",
        ),
        run_id="wba-canary-test",
    )
    primary = service.resolve_run_plan(
        ExperimentRequest(
            experiment_id="wba-transport-ablation-v1",
            preset="primary",
        ),
        run_id="wba-primary-test",
    )

    assert (len(canary.cells), canary.max_workers) == (3, 1)
    assert (len(primary.cells), primary.max_workers) == (48, 1)
    assert len({job.candidate_id for job in canary.jobs}) == 3
    assert {job.model_transport["profile"] for job in canary.jobs} == {
        "responses-proxy",
        "responses-inline",
        "chat-inline",
    }
    assert {job.model_transport["upstream_host"] for job in canary.jobs} == {
        "api.inference.wandb.ai"
    }
    assert {job.model_transport["provider_wire_protocol"] for job in canary.jobs} == {
        "chat_completions"
    }
    assert {
        job.model_transport["profile"]: job.model_transport["bridge_required"]
        for job in canary.jobs
    } == {
        "responses-proxy": True,
        "responses-inline": False,
        "chat-inline": False,
    }
    assert {job.task_id for job in primary.jobs} == {
        "trace-auth-diagnosis",
        "tool-call-reconciliation",
        "evaluation-regression",
        "judge-disagreement",
        "latency-anomaly",
        "retry-cost-anomaly",
        "evidence-intervention",
        "evaluation-plan-artifact",
    }


def test_wba_plan_identity_is_stable_across_task_runtime_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = OperatorService(Path(__file__).parents[1])
    request = ExperimentRequest(
        experiment_id="wba-transport-ablation-v1",
        preset="primary",
    )
    monkeypatch.setattr(job_config, "read_agent_runtime_lock", lambda *args: None)
    monkeypatch.setattr(job_config, "read_task_runtime_lock", lambda *args: None)
    before = service.resolve_run_plan(request, run_id="task-preparation")

    def prepared_lock(manifest: Any, task: Any, repo_root: Path) -> dict[str, Any]:
        del manifest, repo_root
        return {
            "schema_version": 1,
            "task_id": task.id,
            "recipe_sha256": "a" * 64,
            "image": f"fugue-task-{task.id}:prepared",
            "image_id": "sha256:" + "b" * 64,
            "dataset_path": f".fugue/runtime/task-images/{task.id}/dataset",
        }

    monkeypatch.setattr(job_config, "read_task_runtime_lock", prepared_lock)
    monkeypatch.setattr(
        job_config,
        "read_agent_runtime_lock",
        lambda *args: {
            **(job_config.agent_runtime_identity("wba-responses", "amd64") or {}),
            "image_id": "sha256:" + "c" * 64,
            "os": "linux",
        },
    )
    after = service.resolve_run_plan(request, run_id="task-preparation")

    assert [job.candidate_id for job in before.jobs] == [
        job.candidate_id for job in after.jobs
    ]
    assert [job.resolved_candidate.execution_fingerprint for job in before.jobs] == [
        job.resolved_candidate.execution_fingerprint for job in after.jobs
    ]
    assert [cell.id for cell in before.cells] == [cell.id for cell in after.cells]


def test_wba_campaign_exposes_an_exact_queryable_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).parents[1]
    source = campaign_lifecycle.resolve_fugue_source_provenance(repo_root)
    monkeypatch.setattr(
        campaign_lifecycle,
        "resolve_fugue_source_provenance",
        lambda _: {**source, "dirty": False},
    )
    service = CampaignService(repo_root)
    catalog = service.catalog("wba-transport-ablation-v1")
    proposal = build_experiment_proposal(
        proposal_id="wba-transport-qualification-001",
        campaign_id="wba-transport-ablation-v1",
        catalog_digest=catalog.catalog_digest,
        stage_id="qualification",
        research_question="Does transport topology change outcomes?",
        hypothesis="Responses conversion may alter protocol integrity.",
        fixed_dimensions=(
            "model",
            "endpoint",
            "task",
            "system prompt",
            "tool",
            "loop policy",
            "runtime",
            "attempt",
        ),
        varied_dimensions=("transport profile",),
        measured_dimensions=(
            "task pass",
            "artifact pass",
            "protocol integrity",
            "latency",
            "tokens",
            "cost",
        ),
        experiment_id="wba-transport-ablation-v1",
        model="wandb/zai-org/GLM-5.2",
        n_attempts=1,
        n_concurrent=1,
        workloads=("transport",),
        harnesses=("wba-responses",),
        context_systems=("none",),
        variants=("responses-proxy", "responses-inline", "chat-inline"),
        n_tasks=1,
        trace_content="full",
    )

    preview = service.preview(proposal)

    assert preview.cell_count == 3
    assert preview.applicable_cells == 3
    assert preview.expected_predictions == 3
    assert preview.max_concurrent == 1
    assert {cell["model_transport"]["profile"] for cell in preview.cells} == {
        "responses-proxy",
        "responses-inline",
        "chat-inline",
    }
    route_locks = campaign_lifecycle._prepared_route_locks(preview.cells, {})
    assert (
        tuple(campaign_lifecycle._route_lock_from_dict(lock) for lock in route_locks)
        == route_locks
    )


def test_wba_offline_tasks_materialize_and_verify_reference_output(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).parents[1]
    manifest = load_manifest(repo_root / "datasets/wba-transport-ablation-v1.yaml")
    source_path = repo_root / "datasets/wba-transport-ablation-v1.jsonl"
    result = WBATransportTaskMaterializer().materialize(
        manifest,
        tmp_path / "tasks",
        source_path,
        repo_root=repo_root,
    )

    assert result == {"tasks": 8, "offline": True, "scenarios": 4}
    task = tmp_path / "tasks/trace-auth-diagnosis"
    instruction = (task / "instruction.md").read_text()
    assert "/workspace/resources/trace-auth-failure.jsonl" in instruction
    assert "expired service token" not in instruction.casefold()
    interaction = TaskInteractionController.from_environment(
        logs_dir=tmp_path / "interaction",
        initial_instruction=instruction,
        env={
            "FUGUE_TASK_INTERACTION": json.dumps(
                manifest.tasks[0].metadata["interaction_controller"]
            )
        },
    )
    assert interaction.plan.profile_id == "scripted-reviewer-v1"
    assert interaction.plan.follow_up_count == 1

    logs = tmp_path / "logs"
    artifact = logs / "artifacts"
    artifact.mkdir(parents=True)
    (artifact / "fugue-answer.md").write_text(
        (task / "solution/reference-answer.md").read_text()
    )
    (artifact / "root-cause.json").write_text(
        (task / "solution/reference-artifact.json").read_text()
    )
    verifier = tmp_path / "verify.sh"
    verifier.write_text(
        (task / "tests/test.sh").read_text().replace("/logs", logs.as_posix())
    )
    completed = subprocess.run(
        ["sh", verifier.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads((logs / "verifier/reward.json").read_text()) == {
        "answer_facts": 1.0,
        "artifact_contract": 1.0,
        "reward": 1.0,
        "task_pass": 1.0,
    }

    boolean_task = tmp_path / "tasks/judge-disagreement"
    boolean_logs = tmp_path / "boolean-logs"
    boolean_artifacts = boolean_logs / "artifacts"
    boolean_artifacts.mkdir(parents=True)
    (boolean_artifacts / "fugue-answer.md").write_text(
        (boolean_task / "solution/reference-answer.md").read_text()
    )
    (boolean_artifacts / "judge-audit.json").write_text(
        (boolean_task / "solution/reference-artifact.json").read_text()
    )
    boolean_verifier = tmp_path / "verify-booleans.sh"
    boolean_verifier.write_text(
        (boolean_task / "tests/test.sh")
        .read_text()
        .replace("/logs", boolean_logs.as_posix())
    )
    boolean_completed = subprocess.run(
        ["sh", boolean_verifier.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    assert boolean_completed.returncode == 0, boolean_completed.stderr
    assert json.loads((boolean_logs / "verifier/reward.json").read_text()) == {
        "answer_facts": 1.0,
        "artifact_contract": 1.0,
        "reward": 1.0,
        "task_pass": 1.0,
    }


def test_wba_analysis_uses_aligned_task_attempt_contrasts() -> None:
    rows = []
    for task in ("task-a", "task-b"):
        for attempt in (1, 2):
            outcomes = {
                "responses-proxy": False,
                "responses-inline": task == "task-a",
                "chat-inline": task == "task-a" or attempt == 2,
            }
            for profile, passed in outcomes.items():
                rows.append(
                    {
                        "harness": "wba-responses",
                        "task_name": task,
                        "trial_index": attempt,
                        "transport_profile": profile,
                        "pass": passed,
                        "transport_normalization_errors": 0,
                        "transport_stream_anomalies": 0,
                        "transport_stream_anomaly_kinds": {},
                        "transport_orphan_tool_outputs": 0,
                    }
                )

    result = analyze_wba_transport_rows(rows, bootstrap_samples=2_000)

    assert result["complete_grid"] is True
    assert result["aligned_coordinates"] == 4
    assert {
        profile: (value["passes"], value["trials"])
        for profile, value in result["arm_totals"].items()
    } == {
        "responses-proxy": (0, 4),
        "responses-inline": (2, 4),
        "chat-inline": (3, 4),
    }
    assert result["contrasts"][0]["id"] == "refactor_topology"
    assert result["contrasts"][0]["pass_rate_delta"] == 0.5
    assert result["contrasts"][1]["id"] == "responses_stack_gap"
    assert result["contrasts"][1]["discordance"] == {
        "treatment_only_pass": 1,
        "reference_only_pass": 0,
        "same_outcome": 3,
    }


def test_wba_transport_summary_is_required_and_safe_for_campaign_evidence(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "session_id": "session-a",
        "profile": "responses-inline",
        "input_tokens": 10,
        "output_tokens": 4,
        "tool_calls": 2,
        "tool_errors": 0,
        "orphan_tool_outputs": 0,
        "normalization_errors": 0,
        "stream_anomalies": 0,
        "stream_anomaly_kinds": {},
        "stream_events": 9,
        "retries": 0,
        "transport_errors": 0,
        "compactions": 0,
        "stop_reason": "completed",
    }
    (agent / "wba-responses-summary.json").write_text(json.dumps(summary))
    meta = {
        "harness": "wba-responses",
        "model_transport": {"profile": "responses-inline"},
        "native_session_ids": ["session-a"],
    }

    evidence = _wba_transport_evidence(trial, meta)
    safe = safe_prediction_row({"harness": "wba-responses", **evidence})

    assert evidence["wba_transport_status"] == "valid"
    assert safe["transport_profile"] == "responses-inline"
    assert safe["wba_transport"]["tool_calls"] == 2
    assert safe["transport_errors"] == 0
    assert safe["transport_stream_anomalies"] == 0
    normalized_summary = {
        **summary,
        "stream_anomalies": 17,
        "stream_anomaly_kinds": {"reused_output_index": 17},
    }
    (agent / "wba-responses-summary.json").write_text(
        json.dumps(normalized_summary)
    )
    normalized_evidence = _wba_transport_evidence(trial, meta)
    assert normalized_evidence["wba_transport_status"] == "valid"
    assert normalized_evidence["transport_stream_anomalies"] == 17
    assert normalized_evidence["transport_stream_anomaly_kinds"] == {
        "reused_output_index": 17
    }
    recovered_summary = {
        **summary,
        "normalization_errors": 1,
        "transport_errors": 1,
        "retries": 1,
    }
    (agent / "wba-responses-summary.json").write_text(
        json.dumps(recovered_summary)
    )
    assert _wba_transport_evidence(trial, meta)["wba_transport_status"] == "valid"
    failed_summary = {
        **summary,
        "transport_errors": 1,
        "normalization_errors": 1,
        "stop_reason": "transport_error",
    }
    (agent / "wba-responses-summary.json").write_text(json.dumps(failed_summary))
    failed_evidence = _wba_transport_evidence(trial, meta)
    assert failed_evidence["wba_transport_status"] == "invalid"
    assert failed_evidence["transport_errors"] == 1
    (agent / "wba-responses-summary.json").write_text(json.dumps(summary))
    assert (
        _wba_transport_evidence(
            trial,
            {**meta, "native_session_ids": ["different-session"]},
        )["wba_transport_status"]
        == "invalid"
    )
    (agent / "wba-responses-summary.json").unlink()
    assert _wba_transport_evidence(trial, meta)["wba_transport_status"] == "missing"


def test_wba_agent_metadata_includes_transport_error_count() -> None:
    source = Path("fugue/agents/wba_responses.py").read_text()

    assert '"transport_errors": summary.get("transport_errors")' in source
