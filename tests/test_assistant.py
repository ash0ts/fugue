from __future__ import annotations

import asyncio
import json

import httpx

from fugue.assistant import (
    AssistantMessage,
    AssistantModelClient,
    AssistantTool,
    select_assistant_model,
)


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
