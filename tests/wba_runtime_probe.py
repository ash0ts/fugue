"""Networkless probe for the exact pinned WBA runtime image."""

from __future__ import annotations

import asyncio
import json
import os
import runpy
from typing import Any

import httpx
from openai import AsyncOpenAI

os.environ["WANDB_API_KEY"] = "test-only"

RUNNER = runpy.run_path("/opt/fugue-agent-runtime/bin/wba-runner")


class _Events:
    def emit(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _Trace:
    def start_llm(self, *args: Any) -> None:
        del args
        return None

    def finish_llm(self, *args: Any) -> None:
        del args


_CHUNKS = (
    {
        "id": "mock-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "GLM-5.2",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "mock "},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "mock-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "GLM-5.2",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    },
)
_BODY = (
    "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in _CHUNKS) + "data: [DONE]\n\n"
).encode()

_RESPONSE_OUTPUT = [
    {"id": "reason-1", "type": "reasoning", "summary": []},
    {
        "id": "message-1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "mock ok"}],
    },
    {
        "id": "call-1",
        "call_id": "call-1",
        "type": "function_call",
        "name": "shell",
        "arguments": '{"command":"pwd"}',
    },
]
_RESPONSE_EVENTS = (
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
        "item": _RESPONSE_OUTPUT[0],
    },
    {
        "type": "response.output_item.added",
        "sequence_number": 3,
        "output_index": 1,
        "item": {**_RESPONSE_OUTPUT[2], "arguments": ""},
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
        "item": _RESPONSE_OUTPUT[2],
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
        "item": _RESPONSE_OUTPUT[1],
    },
    {
        "type": "response.completed",
        "response": {
            "id": "resp-mock",
            "status": "completed",
            "output": _RESPONSE_OUTPUT,
            "usage": {"input_tokens": 4, "output_tokens": 2},
        },
    },
)
_RESPONSE_BODY = (
    "".join(f"data: {json.dumps(event)}\n\n" for event in _RESPONSE_EVENTS)
    + "data: [DONE]\n\n"
).encode()


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/responses"):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_RESPONSE_BODY,
        )
    assert request.url.path.endswith("/chat/completions"), request.url.path
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_BODY,
    )


async def _probe(profile: str) -> None:
    config = {
        "profile": profile,
        "model_id": "zai-org/GLM-5.2",
        "litellm_model": "nebius/zai-org/GLM-5.2",
        "provider_key_env": "WANDB_API_KEY",
        "provider_base_url": "http://mock/v1",
        "provider_headers": {},
        "bridge_key_env": "LITELLM_MASTER_KEY",
        "bridge_base_url": "http://mock/v1",
        "system_prompt": "fixed",
    }
    os.environ["LITELLM_MASTER_KEY"] = "test-only"
    client = RUNNER["ModelClient"](config, _Events(), _Trace())
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    openai_client = AsyncOpenAI(
        api_key="test-only",
        base_url="http://mock/v1",
        http_client=http_client,
        max_retries=0,
    )
    if profile == "responses-proxy":
        client._openai = openai_client
    else:
        client._inline_openai = openai_client
    try:
        result = await client.stream([[{"role": "user", "content": "test"}]])
    finally:
        await client.close()
    assert result.text == "mock ok"
    assert result.input_tokens == 4
    assert result.output_tokens == 2
    if profile == "responses-proxy":
        assert [(call.call_id, call.name) for call in result.tool_calls] == [
            ("call-1", "shell")
        ]
        assert client.normalization_errors == 0
        assert client.stream_anomalies > 0
    expected_stop = "stop" if profile == "chat-inline" else "completed"
    assert result.finish_reason == expected_stop


async def _main() -> None:
    await _probe("responses-proxy")
    await _probe("chat-inline")
    await _probe("responses-inline")

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
    assert [(message.role, message.content) for message in turn.messages] == [
        ("assistant", "final answer")
    ]

    import weave

    captured: dict[str, Any] = {}
    original_init = weave.init

    def fake_init(project: str, **kwargs: Any) -> None:
        captured.update(project=project, **kwargs)
        raise RuntimeError("stop after settings capture")

    weave.init = fake_init
    try:
        init_trace = RUNNER["WeaveTrace"](
            {
                "trace_content": "full",
                "weave_project": "wandb/test",
                "conversation_id": "conversation-1",
                "display_model": "wandb/model",
            }
        )
        init_trace.start("instruction")
    finally:
        weave.init = original_init
    assert captured == {
        "project": "wandb/test",
        "settings": {"implicitly_patch_integrations": False},
    }
    original_init(
        "wandb/test",
        settings={"disabled": True, "implicitly_patch_integrations": False},
    )


asyncio.run(_main())
print("networkless WBA transport probe passed")
