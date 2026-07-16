from __future__ import annotations

import asyncio
import json

import httpx
import weave

from fugue.assistant import (
    AssistantAgent,
    AssistantMessage,
    AssistantModelClient,
    AssistantResponse,
    AssistantTool,
    AssistantToolCall,
    AssistantUsage,
    _AssistantTrace,
    select_assistant_model,
)
from fugue.model_plane import resolve_model_route


def test_assistant_model_role_precedence() -> None:
    env = {
        "FUGUE_MODEL": "wandb/default",
        "FUGUE_COMPOSER_MODEL": "openai/composer",
        "FUGUE_ANALYST_MODEL": "anthropic/analyst",
    }

    assert select_assistant_model("composer", env=env) == "openai/composer"
    assert select_assistant_model("analyst", env=env) == "anthropic/analyst"
    assert (
        select_assistant_model(
            "analyst",
            cli_model="wandb/override",
            saved_model="openai/saved",
            experiment_model="openai/experiment",
            env=env,
        )
        == "wandb/override"
    )


def test_provider_clients_normalize_tool_calls_and_usage() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        if request.url.path.endswith("/responses"):
            assert body["tools"][0]["name"] == "submit"
            assert body["tool_choice"] == "required"
            return httpx.Response(
                200,
                json={
                    "id": "response-1",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-openai",
                            "name": "submit",
                            "arguments": '{"value": "openai"}',
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                },
            )
        if request.url.path.endswith("/v1/messages"):
            assert body["tools"][0]["input_schema"]["type"] == "object"
            assert body["tool_choice"] == {"type": "any"}
            return httpx.Response(
                200,
                json={
                    "id": "message-1",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-anthropic",
                            "name": "submit",
                            "input": {"value": "anthropic"},
                        }
                    ],
                    "usage": {"input_tokens": 11, "output_tokens": 4},
                },
            )
        assert request.url.path.endswith("/chat/completions")
        assert body["tools"][0]["function"]["name"] == "submit"
        assert body["tool_choice"] == "required"
        assert request.headers["OpenAI-Project"] == "wandb/fugue-experiments"
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-wandb",
                                    "function": {
                                        "name": "submit",
                                        "arguments": '{"value": "wandb"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handler)
    tool = AssistantTool(
        "submit",
        "Submit a value",
        {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        terminal=True,
    )

    async def exercise() -> None:
        cases = (
            ("openai/gpt-5", {"OPENAI_API_KEY": "secret"}, "openai"),
            ("anthropic/claude", {"ANTHROPIC_API_KEY": "secret"}, "anthropic"),
            ("wandb/model", {"WANDB_API_KEY": "secret"}, "wandb"),
        )
        for model, env, expected in cases:
            client = AssistantModelClient(model, env, transport=transport)
            result = await client.complete(
                [AssistantMessage("user", "test")], tools=[tool]
            )
            assert result.tool_calls[0].arguments == {"value": expected}
            assert result.usage.input_tokens is not None

    asyncio.run(exercise())
    assert len(requests) == 3
    assert all(b"secret" not in request.content for request in requests)


def test_assistant_trace_preserves_weave_conversation_types(monkeypatch) -> None:
    class Usage:
        input_tokens = 0
        output_tokens = 0

    class Span:
        def __init__(self) -> None:
            self.usage = Usage()
            self.input_messages: list[object] = []
            self.output_messages: list[object] = []
            self.closed = False

        def __enter__(self) -> Span:
            return self

        def __exit__(self, *_: object) -> None:
            assert not isinstance(self.usage, dict)
            assert self.usage.input_tokens == 13
            assert self.usage.output_tokens == 5
            assert [message.role for message in self.input_messages] == ["user"]
            assert [message.content for message in self.input_messages] == ["question"]
            assert [message.role for message in self.output_messages] == ["assistant"]
            assert [message.content for message in self.output_messages] == ["done"]
            self.closed = True

    trace = _AssistantTrace(
        role="composer",
        route=resolve_model_route("wandb/test-model", {}),
        env={},
        trace_content="full",
        session_id="session",
        attributes={},
    )
    span = Span()
    trace.turn = object()
    monkeypatch.setattr(weave, "start_llm", lambda **_: span)

    active = trace.start_llm([AssistantMessage("user", "question")])
    trace.finish_llm(
        active,
        response=AssistantResponse(
            text="done",
            usage=AssistantUsage(input_tokens=13, output_tokens=5),
        ),
    )

    assert span.closed is True


def test_assistant_retries_unstructured_response_with_terminal_tool_contract() -> None:
    class Client:
        route = resolve_model_route("wandb/test-model", {})

        def __init__(self) -> None:
            self.calls: list[tuple[AssistantMessage, ...]] = []

        async def complete(
            self,
            messages: list[AssistantMessage],
            *,
            tools: tuple[AssistantTool, ...],
        ) -> AssistantResponse:
            self.calls.append(tuple(messages))
            if len(self.calls) == 1:
                return AssistantResponse(text="I can help with that.")
            assert messages[-1].role == "user"
            assert "submit" in messages[-1].content
            return AssistantResponse(
                text="",
                tool_calls=(
                    AssistantToolCall(
                        id="call-1",
                        name="submit",
                        arguments={"value": "accepted"},
                    ),
                ),
            )

    client = Client()
    agent = AssistantAgent(
        client,  # type: ignore[arg-type]
        role="composer",
        tools=(
            AssistantTool(
                "submit",
                "Submit the result",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                terminal=True,
            ),
        ),
        env={},
        max_rounds=2,
    )

    result = asyncio.run(agent.run([AssistantMessage("user", "compose")]))

    assert result.payload == {"value": "accepted"}
    assert len(client.calls) == 2
