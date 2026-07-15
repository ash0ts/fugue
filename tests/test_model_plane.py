from __future__ import annotations

import pytest

from fugue.model_plane import (
    DEFAULT_MODEL,
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    missing_model_env,
    missing_trace_env,
    provider_client_env,
    provider_request_headers,
    resolve_model_route,
    select_model,
    trace_entity_project,
    trace_env_defaults,
    trace_project_slug,
)
from fugue.weave_support import weave_agents_otel_headers


def test_resolve_model_route_for_supported_providers() -> None:
    wandb = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    assert wandb.provider == "wandb"
    assert wandb.model_id == "zai-org/GLM-5.2"
    assert wandb.api_key_env == "WANDB_API_KEY"
    assert wandb.chat_base_url == "https://api.inference.wandb.ai/v1"
    assert wandb.responses_base_url is None
    assert wandb.messages_base_url is None

    openai = resolve_model_route("openai/gpt-5", {})
    assert openai.provider == "openai"
    assert openai.model_id == "gpt-5"
    assert openai.api_key_env == "OPENAI_API_KEY"
    assert openai.responses_base_url == "https://api.openai.com/v1"

    anthropic = resolve_model_route("anthropic/claude-sonnet-4-5", {})
    assert anthropic.provider == "anthropic"
    assert anthropic.model_id == "claude-sonnet-4-5"
    assert anthropic.api_key_env == "ANTHROPIC_API_KEY"
    assert anthropic.chat_base_url is None
    assert anthropic.messages_base_url == "https://api.anthropic.com"


@pytest.mark.parametrize("model", ["gpt-5", "local/foo", "openai/"])
def test_resolve_model_route_rejects_invalid_models(model: str) -> None:
    with pytest.raises(ValueError):
        resolve_model_route(model, {})


def test_select_model_precedence() -> None:
    env = {"FUGUE_MODEL": "anthropic/env-model"}
    assert select_model("openai/cli", "wandb/manifest", env) == "openai/cli"
    assert select_model(None, "wandb/manifest", env) == "wandb/manifest"
    assert select_model(None, None, env) == "anthropic/env-model"
    assert select_model(None, None, {}) == DEFAULT_MODEL
    assert (
        select_model(
            None,
            "wandb/manifest",
            env,
            harness_model="openai/harness",
            experiment_model="anthropic/experiment",
        )
        == "openai/harness"
    )
    assert (
        select_model(
            None,
            "wandb/manifest",
            env,
            experiment_model="anthropic/experiment",
        )
        == "anthropic/experiment"
    )


def test_missing_env_separates_model_and_trace_keys() -> None:
    route = resolve_model_route("openai/gpt-5", {})
    env = {"WANDB_API_KEY": "trace"}
    assert missing_trace_env(env) == []
    assert missing_model_env(route, env) == ["OPENAI_API_KEY"]


def test_trace_project_defaults_to_wandb_shared_project() -> None:
    assert trace_entity_project({}) == (DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT)
    assert trace_project_slug({}) == "wandb/fugue-experiments"
    assert trace_env_defaults({}) == {
        "WANDB_ENTITY": "wandb",
        "WANDB_PROJECT": "fugue-experiments",
        "WEAVE_PROJECT": "wandb/fugue-experiments",
    }
    assert trace_entity_project({"WEAVE_PROJECT": "custom/project"}) == (
        "custom",
        "project",
    )


def test_wandb_model_requests_use_the_trace_project_for_billing() -> None:
    wandb = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    env = {"WEAVE_PROJECT": "team/experiment-project"}

    assert provider_request_headers(wandb, env) == {
        "OpenAI-Project": "team/experiment-project"
    }
    assert provider_client_env(wandb, env) == {
        "OPENAI_PROJECT": "team/experiment-project",
        "OPENAI_PROJECT_ID": "team/experiment-project",
    }

    openai = resolve_model_route("openai/gpt-5", {})
    assert provider_request_headers(openai, env) == {}
    assert provider_client_env(openai, env) == {}


def test_weave_agents_otel_headers_route_without_exposing_plain_key() -> None:
    headers = weave_agents_otel_headers("wandb/fugue-experiments", "test-key")

    assert headers.startswith("project_id=wandb/fugue-experiments,Authorization=Basic%20")
    assert "test-key" not in headers
